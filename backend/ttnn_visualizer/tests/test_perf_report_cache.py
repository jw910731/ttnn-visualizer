# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The perf report route reuses a finished report, and no build leaves a figure behind.

On a real capture a build takes seconds and about a gigabyte of peak memory, and the
performance page asks again on every filter toggle and remount. tt-perf-report also
drew one ~18 MB pyplot figure per build and never closed it. These tests pin what the
cache may reuse, what it must not, and that pyplot is left headless and empty.
"""

import os
import shutil
from http import HTTPStatus
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import orjson
import pytest
from tt_perf_report import perf_report
from ttnn_visualizer import csv_queries
from ttnn_visualizer.csv_queries import (
    PERF_REPORT_CACHE_MAX_BYTES,
    PERF_REPORT_CACHE_MAX_ENTRIES,
    OpsPerformanceReportQueries,
    _BoundedBytesCache,
)
from ttnn_visualizer.exceptions import DataFormatError
from ttnn_visualizer.extensions import db
from ttnn_visualizer.models import Instance, InstanceTable

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_CSV = (
    REPO_ROOT
    / "scripts"
    / "fixtures"
    / "smoke-performance-report"
    / "ops_perf_results.csv"
)
CAPTURE_FILE_NAME = "ops_perf_results.csv"
REPORT_ENDPOINT = "/api/performance/perf-results/report"
GENERATE_PERF_REPORT = "ttnn_visualizer.csv_queries.perf_report.generate_perf_report"

# What the view forwards for a request that sets nothing, so a direct build can be
# compared byte for byte with what the route serves.
VIEW_DEFAULT_KWARGS = {
    "start_signpost": None,
    "end_signpost": None,
    "print_signposts": True,
    "hide_host_ops": True,
    "merge_devices": True,
    "tracing_mode": False,
    "group_by": None,
}

# One value per argument the view forwards, each different from that view default.
NON_DEFAULT_QUERY_BY_PARAM = {
    "start_signpost": "signpost-a",
    "end_signpost": "signpost-b",
    "print_signposts": "false",
    "hide_host_ops": "false",
    "merge_devices": "false",
    "tracing_mode": "true",
    "group_by": "category",
}

# The steps a build reads back best-effort: a failure in one is logged and the report
# served with that part empty, rather than failing the request.
BEST_EFFORT_STEPS = [
    # The raw capture: signposts and every pass-through column.
    "extract_signposts",
    # tt-perf-report's per-op rows.
    "_resolve_report_column_indexes",
    # tt-perf-report's stacked rows.
    "set_op_type_from_signposts",
]


def _register_instance(app, instance_id, performance_path):
    with app.app_context():
        db.session.add(
            InstanceTable(
                instance_id=instance_id,
                active_report={},
                performance_path=str(performance_path),
            )
        )
        db.session.commit()
    return instance_id


def _get_report(client, instance_id, **query):
    return client.get(
        REPORT_ENDPOINT, query_string={"instanceId": instance_id, **query}
    )


def _fail_once(monkeypatch, step):
    """Make one build step raise on its first call only, as a transient failure would."""
    original = getattr(OpsPerformanceReportQueries, step)
    failed = False

    def _flaky(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise MemoryError(f"{step} could not allocate")
        return original(*args, **kwargs)

    monkeypatch.setattr(OpsPerformanceReportQueries, step, staticmethod(_flaky))


def _serialised_direct_build(performance_path):
    instance = Instance(
        instance_id="pytest-perf-cache-direct", performance_path=str(performance_path)
    )
    return orjson.dumps(
        OpsPerformanceReportQueries.generate_report(instance, **VIEW_DEFAULT_KWARGS)
    )


@pytest.fixture
def capture_dir(tmp_path):
    """A private copy of the smoke capture, so a test can rewrite it in place."""
    directory = tmp_path / "reports" / "capture-a"
    directory.mkdir(parents=True)
    shutil.copy(FIXTURE_CSV, directory / CAPTURE_FILE_NAME)
    return directory


@pytest.fixture
def capture_instance(app, capture_dir):
    return _register_instance(app, "pytest-perf-cache", capture_dir)


@pytest.fixture
def build_spy():
    """Counts tt-perf-report runs while still producing the real report."""
    with mock.patch(
        GENERATE_PERF_REPORT, wraps=perf_report.generate_perf_report
    ) as spy:
        yield spy


@pytest.fixture
def stub_build():
    """Counts builds without running tt-perf-report, for tests about the key alone.

    The route still answers 200, with empty rows: the view reads back whatever the
    library wrote, and this writes nothing. That keeps an argument tt-perf-report
    would reject, such as a signpost the capture lacks, from failing the build and
    passing as a miss for the wrong reason.
    """
    with mock.patch(GENERATE_PERF_REPORT, return_value=None) as stub:
        yield stub


def test_pyplot_is_pinned_to_a_non_gui_backend():
    """A forked gunicorn worker on macOS must never start the Cocoa backend."""
    assert plt.get_backend().lower() == "agg"


def test_a_build_leaves_no_figure_open(capture_dir):
    instance = Instance(
        instance_id="pytest-perf-cache-figures", performance_path=str(capture_dir)
    )

    with mock.patch.object(
        perf_report, "plot_stacked_report", wraps=perf_report.plot_stacked_report
    ) as plot:
        OpsPerformanceReportQueries.generate_report(instance)

    # The library only draws when a stacked report comes out, so a build that never
    # drew would pass the figure check below without proving anything.
    assert plot.called
    assert plt.get_fignums() == []


def test_a_build_that_fails_after_drawing_still_closes_the_figure(capture_dir):
    instance = Instance(
        instance_id="pytest-perf-cache-failed-figure",
        performance_path=str(capture_dir),
    )

    def _draw_then_fail(*args, **kwargs):
        plt.figure()
        raise ValueError("tt-perf-report failed after drawing")

    with mock.patch(GENERATE_PERF_REPORT, side_effect=_draw_then_fail):
        with pytest.raises(DataFormatError):
            OpsPerformanceReportQueries.generate_report(instance)

    assert plt.get_fignums() == []


def test_a_repeated_request_is_served_without_rebuilding(
    client, capture_dir, capture_instance, build_spy
):
    first = _get_report(client, capture_instance)
    second = _get_report(client, capture_instance)

    assert first.status_code == second.status_code == HTTPStatus.OK
    assert build_spy.call_count == 1
    assert second.mimetype == "application/json"
    # Byte for byte what an uncached build serialises, not merely an equal report.
    assert second.get_data() == first.get_data()
    assert second.get_data() == _serialised_direct_build(capture_dir)


@pytest.mark.parametrize("param", sorted(NON_DEFAULT_QUERY_BY_PARAM))
def test_every_report_argument_is_part_of_the_key(
    param, client, capture_instance, stub_build, perf_report_cache
):
    varied = {param: NON_DEFAULT_QUERY_BY_PARAM[param]}

    _get_report(client, capture_instance)
    _get_report(client, capture_instance, **varied)
    assert stub_build.call_count == 2

    # Each is then held under its own key, rather than the variant never being held.
    _get_report(client, capture_instance, **varied)
    _get_report(client, capture_instance)
    assert stub_build.call_count == 2
    assert len(perf_report_cache) == 2


def test_a_capture_touched_in_place_is_rebuilt(
    client, capture_dir, capture_instance, stub_build
):
    capture = capture_dir / CAPTURE_FILE_NAME
    _get_report(client, capture_instance)

    before = capture.stat()
    os.utime(capture, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    _get_report(client, capture_instance)

    assert stub_build.call_count == 2


def test_a_capture_that_changed_size_is_rebuilt_even_with_its_mtime_restored(
    client, capture_dir, capture_instance, stub_build
):
    capture = capture_dir / CAPTURE_FILE_NAME
    _get_report(client, capture_instance)

    before = capture.stat()
    with capture.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    os.utime(capture, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert capture.stat().st_mtime_ns == before.st_mtime_ns
    _get_report(client, capture_instance)

    assert stub_build.call_count == 2


def test_a_sibling_named_in_the_request_is_served_its_own_report(
    app, client, capture_dir, capture_instance, build_spy
):
    """`?name=` repoints the instance, so the key has to be taken after it does."""
    app.config["SERVER_MODE"] = False
    sibling = capture_dir.parent / "capture-b"
    sibling.mkdir()
    fixture_lines = FIXTURE_CSV.read_text(encoding="utf-8").splitlines(keepends=True)
    (sibling / CAPTURE_FILE_NAME).write_text(
        "".join(fixture_lines[:6]), encoding="utf-8"
    )

    active = _get_report(client, capture_instance)
    named = _get_report(client, capture_instance, name=sibling.name)

    assert active.status_code == named.status_code == HTTPStatus.OK
    assert build_spy.call_count == 2
    assert named.get_data() != active.get_data()
    assert named.get_data() == _serialised_direct_build(sibling)


def test_a_copy_at_another_path_is_keyed_apart(
    app, client, capture_dir, capture_instance, stub_build
):
    """Same bytes and the same mtime, so only the path tells the two apart."""
    app.config["SERVER_MODE"] = False
    capture = capture_dir / CAPTURE_FILE_NAME
    sibling = capture_dir.parent / "capture-b"
    sibling.mkdir()
    shutil.copy(capture, sibling / CAPTURE_FILE_NAME)
    capture_stat = capture.stat()
    os.utime(
        sibling / CAPTURE_FILE_NAME,
        ns=(capture_stat.st_atime_ns, capture_stat.st_mtime_ns),
    )

    _get_report(client, capture_instance)
    _get_report(client, capture_instance, name=sibling.name)

    assert stub_build.call_count == 2


def test_the_cache_holds_a_bounded_number_of_reports(
    client, capture_instance, stub_build, perf_report_cache
):
    signposts = [
        f"signpost-{index}" for index in range(PERF_REPORT_CACHE_MAX_ENTRIES + 1)
    ]
    for signpost in signposts:
        _get_report(client, capture_instance, start_signpost=signpost)

    assert len(perf_report_cache) == PERF_REPORT_CACHE_MAX_ENTRIES
    builds = stub_build.call_count

    _get_report(client, capture_instance, start_signpost=signposts[-1])
    assert stub_build.call_count == builds

    _get_report(client, capture_instance, start_signpost=signposts[0])
    assert stub_build.call_count == builds + 1


def test_reading_an_entry_protects_it_from_eviction():
    cache = _BoundedBytesCache(max_entries=2, max_bytes=PERF_REPORT_CACHE_MAX_BYTES)
    cache.put("older", b"older")
    cache.put("newer", b"newer")

    assert cache.get("older") == b"older"
    cache.put("newest", b"newest")

    assert cache.get("older") == b"older"
    assert cache.get("newer") is None
    assert len(cache) == 2


def test_the_oldest_reports_are_evicted_to_stay_within_the_byte_budget():
    cache = _BoundedBytesCache(max_entries=4, max_bytes=10)
    cache.put("first", b"1111")
    cache.put("second", b"2222")
    cache.put("third", b"3333")

    assert cache.get("first") is None
    assert cache.get("second") == b"2222"
    assert cache.get("third") == b"3333"
    assert cache.total_bytes == 8


def test_two_builds_racing_to_store_one_key_count_it_once():
    cache = _BoundedBytesCache(max_entries=4, max_bytes=10)
    cache.put("report", b"1111")
    cache.put("report", b"1111")
    cache.put("other", b"2222")

    assert len(cache) == 2
    assert cache.total_bytes == 8


def test_a_report_larger_than_the_byte_budget_is_not_kept():
    cache = _BoundedBytesCache(max_entries=4, max_bytes=10)
    cache.put("small", b"small")
    cache.put("huge", b"x" * 11)

    assert cache.get("huge") is None
    # Nothing is given up to make room for a payload that could never fit.
    assert cache.get("small") == b"small"
    assert cache.total_bytes == len(b"small")


def test_a_report_over_the_byte_budget_is_rebuilt_on_every_request(
    monkeypatch, client, capture_instance, stub_build
):
    budget_below_any_report = _BoundedBytesCache(
        PERF_REPORT_CACHE_MAX_ENTRIES, max_bytes=1
    )
    monkeypatch.setattr(
        csv_queries, "_serialised_report_cache", budget_below_any_report
    )

    first = _get_report(client, capture_instance)
    second = _get_report(client, capture_instance)

    assert first.status_code == second.status_code == HTTPStatus.OK
    assert first.get_data() == second.get_data()
    assert stub_build.call_count == 2
    assert len(budget_below_any_report) == 0


@pytest.mark.parametrize("step", BEST_EFFORT_STEPS)
def test_a_build_that_lost_a_step_is_served_but_not_kept(
    step,
    monkeypatch,
    client,
    capture_dir,
    capture_instance,
    build_spy,
    perf_report_cache,
):
    """A transient failure, such as memory pressure, must clear on the next request.

    Kept, the partial report would be served until the capture changed, which a
    hosted user has no way to make happen.
    """
    _fail_once(monkeypatch, step)

    partial = _get_report(client, capture_instance)
    assert partial.status_code == HTTPStatus.OK
    assert len(perf_report_cache) == 0

    recovered = _get_report(client, capture_instance)
    assert build_spy.call_count == 2
    # Proves the step was reached and emptied part of the first report.
    assert recovered.get_data() != partial.get_data()
    assert recovered.get_data() == _serialised_direct_build(capture_dir)
    assert len(perf_report_cache) == 1


def test_a_failed_build_answers_422_every_time_and_is_not_kept(
    client, capture_instance, perf_report_cache
):
    with mock.patch(
        GENERATE_PERF_REPORT, side_effect=ValueError("unreadable capture")
    ) as failing:
        first = _get_report(client, capture_instance)
        second = _get_report(client, capture_instance)

    assert first.status_code == second.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert first.get_json() == second.get_json() == {"error": "unreadable capture"}
    assert failing.call_count == 2
    assert len(perf_report_cache) == 0

    recovered = _get_report(client, capture_instance)
    assert recovered.status_code == HTTPStatus.OK


def test_a_capture_with_no_rows_answers_422_every_time_and_is_not_kept(
    client, capture_dir, capture_instance, build_spy, perf_report_cache
):
    header = FIXTURE_CSV.read_text(encoding="utf-8").splitlines(keepends=True)[0]
    (capture_dir / CAPTURE_FILE_NAME).write_text(header, encoding="utf-8")

    first = _get_report(client, capture_instance)
    second = _get_report(client, capture_instance)

    assert first.status_code == second.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert build_spy.call_count == 0
    assert len(perf_report_cache) == 0
