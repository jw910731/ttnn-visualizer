# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
API tests for ``GET /api/operations/summary``.

The summary is what every route reads through the footer's range slider and report
link, so it must stay cheap: no tensor, argument or buffer read, whose cost grows
with ``buffers`` (``query_tensors`` alone is ~10 s on an 11M-row report). And the
device-operation names are the input to the memory/performance match, so they must be
every ``function_start`` and ``function_end`` name the graph holds, unfiltered and in
graph order, exactly as the client used to read them out of ``/api/operations``.
"""

import inspect
import json
import logging
import sqlite3
import weakref
from http import HTTPStatus
from typing import Dict, Iterator, List, Tuple
from unittest.mock import patch

import orjson
import pytest
from ttnn_visualizer import views
from ttnn_visualizer.models import DeviceOperation, Operation, StackTrace
from ttnn_visualizer.queries import DatabaseQueries, LocalQueryRunner
from ttnn_visualizer.serializers import serialize_operations_summary
from ttnn_visualizer.tests.report_schemas import SCHEMA_V2

_SUMMARY_URL = "/api/operations/summary"
_SUMMARY_KEYS = {
    "id",
    "name",
    "operation_file_identifier",
    "device_operation_start_names",
    "device_operation_end_names",
}

# Only the three tables the summary reads, each with `rank`, as a multi-host capture
# writes them. Nothing else exists, so a read of any other table fails outright.
_RANKED_SUMMARY_SCHEMA = """
CREATE TABLE operations (
    operation_id int,
    name text,
    duration float,
    rank int NOT NULL DEFAULT 0,
    UNIQUE(operation_id, rank)
);
CREATE TABLE captured_graph (
    operation_id int,
    captured_graph text,
    rank int NOT NULL DEFAULT 0
);
CREATE TABLE stack_traces (
    operation_id int,
    stack_trace text,
    source_file_id int,
    rank int NOT NULL DEFAULT 0
);
"""

# The tables `/api/operations` reads beyond the three the summary needs.
_TABLES_THE_SUMMARY_MUST_NOT_READ = (
    "tensors",
    "buffers",
    "operation_arguments",
    "input_tensors",
    "output_tensors",
    "tensor_producers",
    "tensor_consumers",
    "devices",
)


def _node(node_id, node_type, name=None, *, id_key="id"):
    params = {} if name is None else {"name": name}
    return {id_key: node_id, "node_type": node_type, "params": params}


def _graph(*nodes) -> str:
    return json.dumps(list(nodes))


def _sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _operation_inserts(*operations) -> str:
    """``(id, name)`` rows for the unranked ``SCHEMA_V2`` ``operations`` table."""
    return "\n".join(
        f"INSERT INTO operations VALUES ({op_id}, {_sql_text(name)}, 0.5);"
        for op_id, name in operations
    )


def _graph_insert(operation_id: int, graph) -> str:
    value = "NULL" if graph is None else _sql_text(graph)
    return f"INSERT INTO captured_graph VALUES ({operation_id}, {value});"


def _get_summary(client, instance_id, **query):
    return client.get(_SUMMARY_URL, query_string={"instanceId": instance_id, **query})


def test_summary_lists_every_operation_sorted_by_id_with_only_its_own_fields(
    client, make_report
):
    instance_id = make_report(
        inserts_sql=_operation_inserts(
            (3, "ttnn.softmax"), (1, "ttnn.matmul"), (2, "ttnn.add")
        )
    )

    response = _get_summary(client, instance_id)

    assert response.status_code == HTTPStatus.OK
    assert response.mimetype == "application/json"
    summaries = response.get_json()
    assert [summary["id"] for summary in summaries] == [1, 2, 3]
    assert [summary["name"] for summary in summaries] == [
        "ttnn.matmul",
        "ttnn.add",
        "ttnn.softmax",
    ]
    for summary in summaries:
        assert set(summary) == _SUMMARY_KEYS


def test_summary_is_not_routed_as_an_operation_id(client, make_report):
    """``/operations/<operation_id>`` would 404 on ``summary``, not list anything."""
    instance_id = make_report(inserts_sql=_operation_inserts((1, "ttnn.matmul")))

    response = _get_summary(client, instance_id)

    assert response.status_code == HTTPStatus.OK
    assert isinstance(response.get_json(), list)


def test_summary_requires_an_instance(client):
    assert client.get(_SUMMARY_URL).status_code == HTTPStatus.BAD_REQUEST


def test_device_operation_names_are_every_function_node_unfiltered_in_graph_order(
    client, make_report
):
    """
    The client decides which names are device operations (``isDeviceOperation``), so
    host-side names reach it too, and a repeated name stays repeated: the match is
    positional against the perf report's rows.
    """
    graph = _graph(
        _node(0, "capture_start"),
        _node(1, "function_start", "ttnn.matmul"),
        _node(2, "function_start", "Tensor::to_device"),
        _node(3, "tensor"),
        _node(4, "function_end", "Tensor::to_device"),
        _node(5, "function_start", "MatmulDeviceOperation"),
        _node(6, "buffer_allocate"),
        _node(7, "function_end", "MatmulDeviceOperation"),
        _node(8, "function_start", "MatmulDeviceOperation"),
        _node(9, "function_end", "MatmulDeviceOperation"),
        _node(10, "function_start", "torch.add (torch)"),
        _node(11, "function_start", ""),
        _node(12, "function_end", "ttnn.matmul"),
        _node(13, "capture_end"),
    )
    instance_id = make_report(
        inserts_sql="\n".join(
            [_operation_inserts((1, "ttnn.matmul")), _graph_insert(1, graph)]
        )
    )

    [summary] = _get_summary(client, instance_id).get_json()

    assert summary["device_operation_start_names"] == [
        "ttnn.matmul",
        "Tensor::to_device",
        "MatmulDeviceOperation",
        "MatmulDeviceOperation",
        "torch.add (torch)",
        "",
    ]
    assert summary["device_operation_end_names"] == [
        "Tensor::to_device",
        "MatmulDeviceOperation",
        "MatmulDeviceOperation",
        "ttnn.matmul",
    ]


def test_names_do_not_depend_on_which_node_id_convention_the_report_uses(
    client, make_report
):
    """Older reports name nodes by ``counter`` alone; newer ones write ``id``."""
    legacy = _graph(
        _node(0, "function_start", "AddDeviceOperation", id_key="counter"),
        _node(1, "function_end", "AddDeviceOperation", id_key="counter"),
    )
    current = _graph(
        {**_node(0, "function_start", "AddDeviceOperation"), "counter": 30},
        {**_node(1, "function_end", "AddDeviceOperation"), "counter": 31},
    )
    instance_id = make_report(
        inserts_sql="\n".join(
            [
                _operation_inserts((1, "ttnn.add"), (2, "ttnn.add")),
                _graph_insert(1, legacy),
                _graph_insert(2, current),
            ]
        )
    )

    summaries = _get_summary(client, instance_id).get_json()

    for summary in summaries:
        assert summary["device_operation_start_names"] == ["AddDeviceOperation"]
        assert summary["device_operation_end_names"] == ["AddDeviceOperation"]


def test_a_graph_with_only_function_end_names_keeps_them(client, make_report):
    """
    The client falls back to end order for child-first captures (#1860), so the end
    names must arrive even when no start name does.
    """
    graph = _graph(
        _node(0, "function_end", "UnaryDeviceOperation"),
        _node(1, "function_end", "SparseMatmulDeviceOperation"),
    )
    instance_id = make_report(
        inserts_sql="\n".join(
            [_operation_inserts((59, "ttnn.sparse_matmul")), _graph_insert(59, graph)]
        )
    )

    [summary] = _get_summary(client, instance_id).get_json()

    assert summary["device_operation_start_names"] == []
    assert summary["device_operation_end_names"] == [
        "UnaryDeviceOperation",
        "SparseMatmulDeviceOperation",
    ]


@pytest.mark.parametrize(
    "graph",
    [None, "", "[]", "{}", "null"],
    ids=["null", "empty-string", "empty-array", "object", "json-null"],
)
def test_a_missing_or_empty_graph_lists_no_names(client, make_report, graph):
    instance_id = make_report(
        inserts_sql="\n".join(
            [_operation_inserts((1, "ttnn.add")), _graph_insert(1, graph)]
        )
    )

    [summary] = _get_summary(client, instance_id).get_json()

    assert summary["device_operation_start_names"] == []
    assert summary["device_operation_end_names"] == []


def test_an_operation_without_a_graph_row_lists_no_names(client, make_report):
    instance_id = make_report(
        inserts_sql="\n".join(
            [
                _operation_inserts((1, "ttnn.add"), (2, "ttnn.relu")),
                _graph_insert(1, _graph(_node(0, "function_start", "AddOp"))),
            ]
        )
    )

    summaries = _get_summary(client, instance_id).get_json()

    assert summaries[0]["device_operation_start_names"] == ["AddOp"]
    assert summaries[1]["device_operation_start_names"] == []
    assert summaries[1]["device_operation_end_names"] == []


def test_a_report_without_a_captured_graph_table_still_lists_its_operations(
    client, make_report
):
    schema = SCHEMA_V2.replace(
        "CREATE TABLE captured_graph (\n    operation_id int,\n    captured_graph text\n);",
        "",
    )
    assert "captured_graph" not in schema
    instance_id = make_report(
        schema_sql=schema, inserts_sql=_operation_inserts((1, "ttnn.add"))
    )

    response = _get_summary(client, instance_id)

    assert response.status_code == HTTPStatus.OK
    assert response.get_json() == [
        {
            "id": 1,
            "name": "ttnn.add",
            "operation_file_identifier": "",
            "device_operation_start_names": [],
            "device_operation_end_names": [],
        }
    ]


def test_malformed_nodes_are_skipped_rather_than_failing_the_report(
    client, make_report
):
    graph = json.dumps(
        [
            "not a node",
            {"node_type": "function_start"},
            {"node_type": "function_start", "params": "not an object"},
            {"node_type": "function_start", "params": {"name": 7}},
            {"node_type": "function_end", "params": {"name": None}},
            {"node_type": "function_start", "params": {"name": "AddDeviceOperation"}},
        ]
    )
    instance_id = make_report(
        inserts_sql="\n".join(
            [_operation_inserts((1, "ttnn.add")), _graph_insert(1, graph)]
        )
    )

    [summary] = _get_summary(client, instance_id).get_json()

    assert summary["device_operation_start_names"] == ["AddDeviceOperation"]
    assert summary["device_operation_end_names"] == []


def test_a_graph_that_does_not_parse_costs_its_operation_names_not_the_report(
    client, make_report, caplog
):
    instance_id = make_report(
        inserts_sql="\n".join(
            [
                _operation_inserts((1, "ttnn.add"), (2, "ttnn.relu")),
                _graph_insert(1, '[{"node_type": "function_start"'),
                _graph_insert(2, _graph(_node(0, "function_start", "ReluOp"))),
            ]
        )
    )

    with caplog.at_level(logging.WARNING, logger="ttnn_visualizer.serializers"):
        response = _get_summary(client, instance_id)

    assert response.status_code == HTTPStatus.OK
    summaries = response.get_json()
    assert summaries[0]["device_operation_start_names"] == []
    assert summaries[1]["device_operation_start_names"] == ["ReluOp"]
    assert "1 captured graphs did not parse" in caplog.text


# Graphs orjson refuses but `JSON.parse` reads. The client used to parse every graph
# itself out of `/api/operations`, so each of these listed its names then.
_GRAPHS_ONLY_JSON_PARSE_READS = {
    "lone-surrogate-in-a-param": (
        '[{"node_type": "function_start", "params": '
        '{"name": "MatmulDeviceOperation", "tensor_name": "w\\udc80"}}]'
    ),
    "number-beyond-a-double": (
        '[{"node_type": "function_start", "params": '
        '{"name": "MatmulDeviceOperation", "scale": 1e400}}]'
    ),
    "lone-surrogate-in-the-name": (
        '[{"node_type": "function_start", "params": {"name": "Matmul\\udc80Op"}}, '
        '{"node_type": "function_end", "params": {"name": "Matmul\\udc80Op"}}]'
    ),
}


def _names_from_full_graph(operation, node_type):
    return [
        node["params"]["name"]
        for node in operation["device_operations"]
        if node["node_type"] == node_type
    ]


@pytest.mark.parametrize(
    "graph",
    list(_GRAPHS_ONLY_JSON_PARSE_READS.values()),
    ids=list(_GRAPHS_ONLY_JSON_PARSE_READS),
)
def test_a_graph_orjson_refuses_lists_the_names_json_parse_found_in_the_full_list(
    client, make_report, caplog, graph
):
    """
    Read back with the stdlib parser, which, like ``JSON.parse``, keeps a lone
    surrogate and reads ``1e400`` as infinity; ``/api/operations`` splices the graph
    in verbatim, so its parse is the one the client used to make.
    """
    with pytest.raises(orjson.JSONDecodeError):
        orjson.loads(graph)
    instance_id = make_report(
        inserts_sql="\n".join(
            [_operation_inserts((1, "ttnn.matmul")), _graph_insert(1, graph)]
        )
    )

    with caplog.at_level(logging.WARNING, logger="ttnn_visualizer.serializers"):
        summary_response = _get_summary(client, instance_id)
    full_response = client.get(
        "/api/operations", query_string={"instanceId": instance_id}
    )

    assert summary_response.status_code == HTTPStatus.OK
    assert full_response.status_code == HTTPStatus.OK
    [summary] = json.loads(summary_response.data)
    [operation] = json.loads(full_response.data)
    assert summary["device_operation_start_names"] == _names_from_full_graph(
        operation, "function_start"
    )
    assert summary["device_operation_end_names"] == _names_from_full_graph(
        operation, "function_end"
    )
    assert summary["device_operation_start_names"], "the graph lists a name"
    assert "did not parse" not in caplog.text


def test_a_graph_nested_past_both_parsers_limits_costs_its_names_not_the_report(
    client, make_report, caplog
):
    """
    orjson stops at 1024 levels and the stdlib parser at the recursion limit, 1000 by
    default. ``JSON.parse`` goes deeper, but TTNN's graphs nest three levels.
    """
    too_deep = "[" * 5000 + "]" * 5000
    instance_id = make_report(
        inserts_sql="\n".join(
            [_operation_inserts((1, "ttnn.add")), _graph_insert(1, too_deep)]
        )
    )

    with caplog.at_level(logging.WARNING, logger="ttnn_visualizer.serializers"):
        response = _get_summary(client, instance_id)

    assert response.status_code == HTTPStatus.OK
    assert response.get_json()[0]["device_operation_start_names"] == []
    assert "1 captured graphs did not parse" in caplog.text


@pytest.mark.parametrize(
    "stack_trace_insert,expected",
    [
        (
            "INSERT INTO stack_traces VALUES "
            '(1, \'File "/proj/models/model.py", line 12, in forward\n'
            '  File "/proj/run.py", line 3, in main\');',
            "model.py:12",
        ),
        ("INSERT INTO stack_traces VALUES (1, 'no frames recorded');", ""),
        ("INSERT INTO stack_traces VALUES (1, NULL);", ""),
        ("", ""),
    ],
    ids=["first-frame", "no-frame", "null-trace", "no-trace-row"],
)
def test_file_identifier_is_the_first_frames_file_and_line(
    client, make_report, stack_trace_insert, expected
):
    instance_id = make_report(
        inserts_sql="\n".join(
            [_operation_inserts((1, "ttnn.matmul")), stack_trace_insert]
        )
    )

    [summary] = _get_summary(client, instance_id).get_json()

    assert summary["operation_file_identifier"] == expected


def test_a_report_without_a_stack_traces_table_has_no_file_identifiers(
    client, make_report
):
    schema = SCHEMA_V2.replace(
        "CREATE TABLE stack_traces (\n    operation_id int,\n    stack_trace text\n);",
        "",
    )
    assert "stack_traces" not in schema
    instance_id = make_report(
        schema_sql=schema, inserts_sql=_operation_inserts((1, "ttnn.add"))
    )

    [summary] = _get_summary(client, instance_id).get_json()

    assert summary["operation_file_identifier"] == ""


def _ranked_report_inserts() -> str:
    # Both ranks reuse operation id 1, as the writer does (#1842), so only the rank
    # filter can tell their graphs and traces apart.
    return "\n".join(
        [
            "INSERT INTO operations VALUES (1, 'ttnn.matmul', 1.0, 0), (1, 'ttnn.add', 2.0, 1);",
            "INSERT INTO captured_graph VALUES "
            f"(1, {_sql_text(_graph(_node(0, 'function_start', 'MatmulOp')))}, 0), "
            f"(1, {_sql_text(_graph(_node(0, 'function_start', 'AddOp')))}, 1);",
            "INSERT INTO stack_traces VALUES "
            "(1, 'File \"/r0/zero.py\", line 1, in f', NULL, 0), "
            "(1, 'File \"/r1/one.py\", line 2, in f', NULL, 1);",
        ]
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        ({}, {"name": "ttnn.matmul", "names": ["MatmulOp"], "file": "zero.py:1"}),
        (
            {"rank": "0"},
            {"name": "ttnn.matmul", "names": ["MatmulOp"], "file": "zero.py:1"},
        ),
        ({"rank": "1"}, {"name": "ttnn.add", "names": ["AddOp"], "file": "one.py:2"}),
    ],
    ids=["omitted-means-rank-0", "rank-0", "rank-1"],
)
def test_summary_is_scoped_to_one_rank(client, make_report, query, expected):
    instance_id = make_report(
        schema_sql=_RANKED_SUMMARY_SCHEMA, inserts_sql=_ranked_report_inserts()
    )

    response = _get_summary(client, instance_id, **query)

    assert response.status_code == HTTPStatus.OK
    assert response.get_json() == [
        {
            "id": 1,
            "name": expected["name"],
            "operation_file_identifier": expected["file"],
            "device_operation_start_names": expected["names"],
            "device_operation_end_names": [],
        }
    ]


def test_an_unranked_graph_table_still_pairs_with_ranked_operations(
    client, make_report
):
    """The rank-0 fallback ``/api/operations`` makes for a table without ``rank``."""
    schema = _RANKED_SUMMARY_SCHEMA.replace(
        "    captured_graph text,\n    rank int NOT NULL DEFAULT 0\n",
        "    captured_graph text\n",
    )
    instance_id = make_report(
        schema_sql=schema,
        inserts_sql="\n".join(
            [
                "INSERT INTO operations VALUES (1, 'ttnn.add', 2.0, 1);",
                _graph_insert(1, _graph(_node(0, "function_start", "AddOp"))),
            ]
        ),
    )

    [summary] = _get_summary(client, instance_id, rank="1").get_json()

    assert summary["device_operation_start_names"] == ["AddOp"]


def test_a_nonzero_rank_on_an_unranked_report_is_refused(client, make_report):
    instance_id = make_report(inserts_sql=_operation_inserts((1, "ttnn.add")))

    response = _get_summary(client, instance_id, rank="2")

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "per-rank" in response.get_json()["error"].lower()


def test_an_invalid_rank_is_a_bad_request(client, make_report):
    instance_id = make_report(inserts_sql=_operation_inserts((1, "ttnn.add")))

    assert (
        _get_summary(client, instance_id, rank="-1").status_code
        == HTTPStatus.BAD_REQUEST
    )


def _reject(qualified_name):
    def _raise(*_args, **_kwargs):
        raise AssertionError(f"/operations/summary called {qualified_name}")

    return _raise


def test_summary_never_reads_tensors_arguments_or_buffers(client, make_report):
    """
    Patched on the class, which is where every caller binds these methods, and
    checked again at the SQL the connection actually runs, so a new query that
    reaches one of those tables some other way fails too.
    """
    instance_id = make_report(
        inserts_sql="\n".join(
            [
                _operation_inserts((1, "ttnn.add")),
                _graph_insert(1, _graph(_node(0, "function_start", "AddOp"))),
                "INSERT INTO stack_traces VALUES (1, 'File \"/src/m.py\", line 1, in f');",
            ]
        )
    )
    executed: List[str] = []
    real_execute = LocalQueryRunner.execute_query
    real_iterate = LocalQueryRunner.iterate_query

    def _recording_execute(self, query, params=None):
        executed.append(query)
        return real_execute(self, query, params)

    def _recording_iterate(self, query, params=None):
        executed.append(query)
        return real_iterate(self, query, params)

    guarded = [
        "query_tensors",
        "query_input_tensors",
        "query_output_tensors",
        "query_operation_arguments",
        "query_buffers",
        "query_producers_consumers",
        "query_devices",
        "query_buffer_totals_by_operation",
    ]
    patches = [
        patch.object(DatabaseQueries, name, _reject(f"DatabaseQueries.{name}"))
        for name in guarded
    ]
    patches += [
        patch.object(LocalQueryRunner, "execute_query", _recording_execute),
        patch.object(LocalQueryRunner, "iterate_query", _recording_iterate),
    ]
    for active in patches:
        active.start()
    try:
        response = _get_summary(client, instance_id)
    finally:
        for active in reversed(patches):
            active.stop()

    assert response.status_code == HTTPStatus.OK
    assert response.get_json()[0]["device_operation_start_names"] == ["AddOp"]
    selects = [
        query for query in executed if query.lstrip().upper().startswith("SELECT")
    ]
    assert selects, "the recorder saw no queries, so it proves nothing"
    for query in selects:
        for table in _TABLES_THE_SUMMARY_MUST_NOT_READ:
            assert f"FROM {table} " not in f"{query} ", query


def test_summary_serves_a_report_holding_only_the_tables_it_reads(client, make_report):
    """A read of any other table would fail with ``no such table`` here."""
    instance_id = make_report(
        schema_sql=_RANKED_SUMMARY_SCHEMA, inserts_sql=_ranked_report_inserts()
    )

    response = _get_summary(client, instance_id)

    assert response.status_code == HTTPStatus.OK
    assert [summary["id"] for summary in response.get_json()] == [1]


def test_graphs_are_streamed_rather_than_fetched_whole(tmp_path):
    """
    ``captured_graph`` is the largest column the summary reads (240 MB on an
    18,740-operation report), so it must not go through ``fetchall``.
    """
    path = tmp_path / "report.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(_RANKED_SUMMARY_SCHEMA + _ranked_report_inserts())
    connection.commit()
    try:
        queries = DatabaseQueries(connection=connection)
        # Probed first, so the schema is cached and only the row read runs below.
        queries._check_table_exists("captured_graph")
        queries._get_table_columns("captured_graph")
        with patch.object(
            LocalQueryRunner,
            "execute_query",
            _reject("LocalQueryRunner.execute_query for a graph"),
        ):
            graphs = list(queries.stream_device_operations({"rank": 1}))
    finally:
        connection.close()

    assert [graph.operation_id for graph in graphs] == [1]
    assert json.loads(graphs[0].captured_graph)[0]["params"]["name"] == "AddOp"


def test_stack_traces_are_streamed_rather_than_fetched_whole(tmp_path):
    """Traces average 7 KB, 70 MB of them on the largest local report."""
    path = tmp_path / "report.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(_RANKED_SUMMARY_SCHEMA + _ranked_report_inserts())
    connection.commit()
    try:
        queries = DatabaseQueries(connection=connection)
        queries._check_table_exists("stack_traces")
        queries._get_table_columns("stack_traces")
        with patch.object(
            LocalQueryRunner,
            "execute_query",
            _reject("LocalQueryRunner.execute_query for a stack trace"),
        ):
            traces = list(queries.stream_stack_traces({"rank": 1}))
    finally:
        connection.close()

    assert [trace.stack_trace for trace in traces] == [
        'File "/r1/one.py", line 2, in f'
    ]


def test_the_route_hands_the_serializer_unread_streams(client, make_report):
    """
    The query methods streaming is not enough: a route that fetched every graph, or
    read them through a method that does, would hold the whole column again -- 250 MiB
    more peak memory per request on the 18,740-operation report. So the serializer must
    get generators nothing has started, and every row read of the two large columns
    must go through ``iterate_query``.
    """
    instance_id = make_report(
        inserts_sql="\n".join(
            [
                _operation_inserts((1, "ttnn.add")),
                _graph_insert(1, _graph(_node(0, "function_start", "AddOp"))),
                "INSERT INTO stack_traces VALUES (1, 'File \"/src/m.py\", line 1, in f');",
            ]
        )
    )
    reads: List[Tuple[str, str]] = []
    real_execute = LocalQueryRunner.execute_query
    real_iterate = LocalQueryRunner.iterate_query

    def _recording_execute(self, query, params=None):
        reads.append(("execute_query", query))
        return real_execute(self, query, params)

    def _recording_iterate(self, query, params=None):
        reads.append(("iterate_query", query))
        return real_iterate(self, query, params)

    stream_states: Dict[str, str] = {}

    def _inspecting_serializer(operations, device_operations, stack_traces):
        for name, stream in (
            ("device_operations", device_operations),
            ("stack_traces", stack_traces),
        ):
            stream_states[name] = (
                inspect.getgeneratorstate(stream)
                if inspect.isgenerator(stream)
                else type(stream).__name__
            )
        return serialize_operations_summary(operations, device_operations, stack_traces)

    with (
        patch.object(LocalQueryRunner, "execute_query", _recording_execute),
        patch.object(LocalQueryRunner, "iterate_query", _recording_iterate),
        patch.object(views, "serialize_operations_summary", _inspecting_serializer),
    ):
        response = _get_summary(client, instance_id)

    assert response.status_code == HTTPStatus.OK
    assert response.get_json()[0]["operation_file_identifier"] == "m.py:1"
    assert stream_states == {
        "device_operations": inspect.GEN_CREATED,
        "stack_traces": inspect.GEN_CREATED,
    }
    for table in ("captured_graph", "stack_traces"):
        methods = {method for method, query in reads if f"FROM {table} " in f"{query} "}
        assert methods == {"iterate_query"}, (table, methods)


def _tracked(items, refs):
    """Yields ``items``, failing if the consumer kept any but the one it holds."""
    for item in items:
        refs.append(weakref.ref(item))
        yield item
        del item
        # The consumer's loop variable still holds the item it just reduced.
        assert sum(ref() is not None for ref in refs) <= 1, "an earlier row was kept"


def test_the_serializer_reduces_each_graph_and_trace_before_reading_the_next():
    """What makes the streams worth having: no two rows are ever held at once."""
    operation_ids = range(1, 6)
    graph_refs: List[weakref.ref] = []
    trace_refs: List[weakref.ref] = []

    def _graphs() -> Iterator[DeviceOperation]:
        for operation_id in operation_ids:
            yield DeviceOperation(
                operation_id, _graph(_node(0, "function_start", f"Op{operation_id}"))
            )

    def _traces() -> Iterator[StackTrace]:
        for operation_id in operation_ids:
            yield StackTrace(
                operation_id, f'File "/src/m{operation_id}.py", line 1, in f', None
            )

    summaries = serialize_operations_summary(
        [Operation(operation_id, "ttnn.op", 1.0) for operation_id in operation_ids],
        _tracked(_graphs(), graph_refs),
        _tracked(_traces(), trace_refs),
    )

    assert len(graph_refs) == len(trace_refs) == len(operation_ids)
    assert [summary["device_operation_start_names"] for summary in summaries] == [
        [f"Op{operation_id}"] for operation_id in operation_ids
    ]
    assert [summary["operation_file_identifier"] for summary in summaries] == [
        f"m{operation_id}.py:1" for operation_id in operation_ids
    ]
