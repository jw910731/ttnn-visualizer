# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import dataclasses
import json
import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple, Union

import orjson
from ttnn_visualizer.models import (
    BufferType,
    DeviceOperation,
    Operation,
    StackTrace,
    TensorComparisonRecord,
)
from ttnn_visualizer.operation_file_identifier import operation_file_identifier

logger = logging.getLogger(__name__)

_EMPTY_DEVICE_OPERATIONS = orjson.Fragment(b"[]")

_FUNCTION_START = "function_start"
_FUNCTION_END = "function_end"

# A device-operation name as the summary writes it: the string, or -- for one holding
# a lone surrogate, which orjson will not write -- that string already written as JSON.
_DeviceOperationName = Union[str, orjson.Fragment]


def _captured_graph_fragment(captured_graph):
    """Wrap a raw captured_graph JSON string so orjson splices it in as-is."""
    if isinstance(captured_graph, orjson.Fragment):
        return captured_graph
    if not captured_graph:
        return _EMPTY_DEVICE_OPERATIONS
    return orjson.Fragment(captured_graph)


def _stack_trace_for_operation(stack_traces_by_key, operation):
    key = (operation.operation_id, operation.rank)
    if key in stack_traces_by_key:
        return stack_traces_by_key[key]
    return stack_traces_by_key.get((operation.operation_id, 0))


def _stack_trace_source_file_id_for_operation(
    stack_trace_source_file_ids_by_key, operation
):
    key = (operation.operation_id, operation.rank)
    if key in stack_trace_source_file_ids_by_key:
        return stack_trace_source_file_ids_by_key[key]
    return stack_trace_source_file_ids_by_key.get((operation.operation_id, 0))


def _error_for_operation(errors_by_key, operation):
    key = (operation.operation_id, operation.rank)
    if key in errors_by_key:
        return errors_by_key[key]
    return errors_by_key.get((operation.operation_id, 0))


def _device_ops_for_operation(device_ops_by_key, operation):
    key = (operation.operation_id, operation.rank)
    if key in device_ops_by_key:
        return device_ops_by_key[key]
    return device_ops_by_key.get((operation.operation_id, 0), _EMPTY_DEVICE_OPERATIONS)


def serialize_operations(
    inputs,
    operation_arguments,
    operations,
    outputs,
    stack_traces,
    tensors,
    devices,
    producers_consumers,
    device_operations,
    error_records=None,
):
    tensors_dict = {(t.tensor_id, t.rank): t for t in tensors}
    device_operations_dict = {
        (do.operation_id, do.rank): _captured_graph_fragment(do.captured_graph)
        for do in device_operations
        if hasattr(do, "operation_id")
    }

    stack_traces_dict = {
        (st.operation_id, st.rank): st.stack_trace for st in stack_traces
    }
    stack_trace_source_file_ids_dict = {
        (st.operation_id, st.rank): st.source_file_id for st in stack_traces
    }

    errors_dict = {}
    if error_records:
        for error in error_records:
            errors_dict[(error.operation_id, error.rank)] = error.to_nested_dict()

    arguments_dict = defaultdict(list)
    for argument in operation_arguments:
        arguments_dict[argument.operation_id].append(argument)

    inputs_dict, outputs_dict = serialize_inputs_outputs(
        inputs, outputs, producers_consumers, tensors_dict
    )

    results = []
    for operation in operations:
        inputs = inputs_dict[operation.operation_id]
        outputs = outputs_dict[operation.operation_id]
        arguments = [a.to_dict() for a in arguments_dict[operation.operation_id]]
        operation_data = operation.to_dict()
        operation_data["id"] = operation.operation_id
        operation_device_operations = _device_ops_for_operation(
            device_operations_dict, operation
        )
        id = operation_data.pop("operation_id", None)

        error_data = _error_for_operation(errors_dict, operation)

        results.append(
            {
                **operation_data,
                "id": id,
                "stack_trace": _stack_trace_for_operation(stack_traces_dict, operation),
                "stack_trace_source_file_id": _stack_trace_source_file_id_for_operation(
                    stack_trace_source_file_ids_dict, operation
                ),
                "device_operations": operation_device_operations,
                "arguments": arguments,
                "inputs": inputs,
                "outputs": outputs,
                "error": error_data,
            }
        )
    return results


def _device_operation_names(
    captured_graph: str,
) -> Tuple[List[_DeviceOperationName], List[_DeviceOperationName]]:
    """
    ``params.name`` of every ``function_start`` and every ``function_end`` node,
    each in graph order. Raises ``ValueError`` for a graph that is not JSON, and
    ``RecursionError`` for one nested past the interpreter's recursion limit --
    which ``JSON.parse`` would read, but TTNN's graphs nest three levels deep.

    Unfiltered on purpose: which names are device operations is
    ``isDeviceOperation``'s decision (``src/functions/filterOperations.ts``), and
    the client applies it to these lists as it does to the full graph, so the
    rule has one home. Only names that are not strings are dropped, since the
    client could do nothing with them but fail.
    """
    try:
        nodes = orjson.loads(captured_graph)
        read_by_stdlib = False
    except orjson.JSONDecodeError:
        # The client read these graphs itself, with `JSON.parse`, when it loaded the
        # full list, and that accepts what orjson refuses: a lone-surrogate escape
        # (`"\udc80"`) and a number beyond a double's range (`1e400`). The stdlib
        # parser takes both, so such a graph keeps the names the client used to
        # find in it. Only a graph orjson refused pays for the slower parse.
        nodes = json.loads(captured_graph)
        read_by_stdlib = True

    start_names: List[_DeviceOperationName] = []
    end_names: List[_DeviceOperationName] = []
    if not isinstance(nodes, list):
        return start_names, end_names
    for node in nodes:
        if not isinstance(node, dict):
            continue
        params = node.get("params")
        name = params.get("name") if isinstance(params, dict) else None
        if not isinstance(name, str):
            continue
        written_name = _json_writable_name(name) if read_by_stdlib else name
        node_type = node.get("node_type")
        if node_type == _FUNCTION_START:
            start_names.append(written_name)
        elif node_type == _FUNCTION_END:
            end_names.append(written_name)
    return start_names, end_names


def _json_writable_name(name: str) -> _DeviceOperationName:
    # orjson will not write a lone surrogate, and only the stdlib parser lets one
    # through. The stdlib writer escapes it (`"\udc80"`), which `JSON.parse` reads
    # back as the same string the client found in the full graph.
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return orjson.Fragment(json.dumps(name))
    return name


def _value_for_operation(values_by_key, operation, default=None):
    # The same (id, rank)-then-rank-0 lookup `serialize_operations` makes, so a
    # table without a `rank` column still pairs with a ranked `operations`.
    key = (operation.operation_id, operation.rank)
    if key in values_by_key:
        return values_by_key[key]
    return values_by_key.get((operation.operation_id, 0), default)


def serialize_operations_summary(
    operations: Iterable[Operation],
    device_operations: Iterable[DeviceOperation],
    stack_traces: Iterable[StackTrace],
) -> List[Dict[str, Any]]:
    """
    The fields of ``/api/operations`` that the performance route, the range
    slider and the report link read, one entry per operation in ``operations``
    order.

    ``device_operations`` and ``stack_traces`` are consumed once each and may be
    streams: every graph and trace is reduced as it arrives, so neither column
    is ever resident in full.
    """
    names_by_key: Dict[
        Tuple[int, int],
        Tuple[List[_DeviceOperationName], List[_DeviceOperationName]],
    ] = {}
    unreadable_graph_count = 0
    for device_operation in device_operations:
        try:
            names_by_key[(device_operation.operation_id, device_operation.rank)] = (
                _device_operation_names(device_operation.captured_graph)
            )
        # Both parsers raise a `ValueError` for text that is not JSON; the stdlib
        # one raises `RecursionError` for nesting past the interpreter's limit.
        except (ValueError, RecursionError):
            unreadable_graph_count += 1

    if unreadable_graph_count:
        # Once per request rather than per graph: a damaged capture can hold
        # thousands, and every one has the same consequence.
        logger.warning(
            "%d captured graphs did not parse as JSON; their operations list no "
            "device operations",
            unreadable_graph_count,
        )

    identifiers_by_key = {
        (stack_trace.operation_id, stack_trace.rank): operation_file_identifier(
            stack_trace.stack_trace
        )
        for stack_trace in stack_traces
    }

    results = []
    for operation in operations:
        names = _value_for_operation(names_by_key, operation)
        start_names, end_names = names if names is not None else ([], [])
        results.append(
            {
                "id": operation.operation_id,
                "name": operation.name,
                "operation_file_identifier": _value_for_operation(
                    identifiers_by_key, operation, ""
                ),
                "device_operation_start_names": start_names,
                "device_operation_end_names": end_names,
            }
        )
    return results


def serialize_inputs_outputs(
    inputs,
    outputs,
    producers_consumers,
    tensors_dict,
    comparisons=None,
):
    producers_consumers_dict = {
        (pc.tensor_id, pc.rank): pc for pc in producers_consumers
    }

    def attach_tensor_data(values):
        values_dict = defaultdict(list)
        for value in values:
            tensor = tensors_dict.get((value.tensor_id, value.rank))
            if tensor is None:
                tensor = tensors_dict.get((value.tensor_id, 0))
            if tensor is None:
                continue
            tensor_dict = tensor.to_dict()
            pc = producers_consumers_dict.get((value.tensor_id, value.rank))
            if pc is None:
                pc = producers_consumers_dict.get((value.tensor_id, 0))
            value_dict = dataclasses.asdict(value)
            value_dict.pop("tensor_id", None)
            value_dict.update(
                {
                    "id": tensor_dict.pop("tensor_id"),
                    "consumers": pc.consumers if pc else [],
                    "producers": pc.producers if pc else [],
                }
            )
            if comparisons:
                comparison = comparisons.get(value.tensor_id)
                value_dict.update({"comparison": comparison})

            values_dict[value.operation_id].append({**value_dict, **tensor_dict})
        return values_dict

    inputs_dict = attach_tensor_data(inputs)
    outputs_dict = attach_tensor_data(outputs)
    return inputs_dict, outputs_dict


def serialize_buffer_pages(buffer_pages):
    # Collect device-specific data if needed

    # Serialize each buffer page to a dictionary using dataclasses.asdict
    buffer_pages_list = [page.to_dict() for page in buffer_pages]

    # Optionally, modify or adjust the serialized data as needed
    for page_data in buffer_pages_list:
        # Set a custom id field if needed
        page_data["id"] = f"{page_data['operation_id']}_{page_data['page_index']}"

        # If the buffer_type is handled by an enum, adjust it similarly to your BufferPage model
        if "buffer_type" in page_data and isinstance(
            page_data["buffer_type"], BufferType
        ):
            page_data["buffer_type"] = page_data["buffer_type"].value

    return buffer_pages_list


def serialize_buffer_chunks(buffer_chunks):
    """
    Serialize ``BufferChunk`` rows for the ``/api/buffer-pages`` response.

    The synthetic ``id`` encodes the full GROUP BY tuple — ``operation_id``,
    ``device_id``, ``address``, ``bank_id``, ``core_x``, ``core_y``,
    ``buffer_type``, ``rank`` — so rows from multi-device / mixed-rank /
    mixed-type responses cannot collide on the same id. ``BufferChunk.to_dict``
    already unwraps the ``buffer_type`` enum to its int value, so the f-string
    embeds the same scalar the client sees.
    """
    result = [chunk.to_dict() for chunk in buffer_chunks]
    for chunk in result:
        chunk["id"] = (
            f"{chunk['operation_id']}_{chunk['device_id']}_{chunk['address']}_"
            f"{chunk['bank_id']}_{chunk['core_x']}_{chunk['core_y']}_"
            f"{chunk['buffer_type']}_{chunk['rank']}"
        )
    return result


def comparisons_by_tensor_id(
    local_comparisons: List[TensorComparisonRecord],
    global_comparisons: List[TensorComparisonRecord],
) -> DefaultDict[int, dict[str, TensorComparisonRecord]]:
    comparisons: DefaultDict[int, dict[str, TensorComparisonRecord]] = defaultdict(dict)
    for local_comparison in local_comparisons:
        comparisons[local_comparison.tensor_id].update({"local": local_comparison})
    for global_comparison in global_comparisons:
        comparisons[global_comparison.tensor_id].update({"global": global_comparison})
    return comparisons


def serialize_operation(
    buffers,
    inputs,
    operation,
    operation_arguments,
    outputs,
    stack_trace,
    tensors,
    global_tensor_comparisons,
    local_tensor_comparisons,
    devices,
    producers_consumers,
    device_operations,
    error_record=None,
):
    tensors_dict = {(t.tensor_id, t.rank): t for t in tensors}
    comparisons = comparisons_by_tensor_id(
        local_tensor_comparisons, global_tensor_comparisons
    )

    inputs_dict, outputs_dict = serialize_inputs_outputs(
        inputs,
        outputs,
        producers_consumers,
        tensors_dict,
        comparisons,
    )

    buffer_list = [buffer.to_dict() for buffer in buffers]

    l1_sizes = [d.worker_l1_size for d in devices]
    arguments_data = [argument.to_dict() for argument in operation_arguments]
    operation_data = operation.to_dict()
    operation_data["id"] = operation.operation_id

    inputs_data = inputs_dict.get(operation.operation_id)
    outputs_data = outputs_dict.get(operation.operation_id)

    id = operation_data.pop("operation_id", None)

    device_operations_data = _EMPTY_DEVICE_OPERATIONS
    chosen = False
    for do in device_operations:
        if do.operation_id == operation.operation_id and do.rank == operation.rank:
            device_operations_data = _captured_graph_fragment(do.captured_graph)
            chosen = True
            break
    if not chosen:
        for do in device_operations:
            if do.operation_id == operation.operation_id:
                device_operations_data = _captured_graph_fragment(do.captured_graph)
                break

    # Convert error record to nested dict if it exists (excludes operation_id and operation_name)
    error_data = error_record.to_nested_dict() if error_record else None

    return {
        **operation_data,
        "id": id,
        "l1_sizes": l1_sizes,
        "device_operations": device_operations_data,
        "stack_trace": stack_trace.stack_trace if stack_trace else "",
        "stack_trace_source_file_id": (
            stack_trace.source_file_id
            if stack_trace and stack_trace.source_file_id is not None
            else None
        ),
        "buffers": buffer_list,
        "arguments": arguments_data,
        "inputs": inputs_data or [],
        "outputs": outputs_data or [],
        "error": error_data,
    }


def serialize_operation_buffers(operation: Operation, operation_buffers):
    buffer_data = []
    for b in operation_buffers:
        buffer_dict = {
            "device_id": b.device_id,
            "address": b.address,
            "buffer_type": (
                b.buffer_type.value
                if hasattr(b.buffer_type, "value")
                else b.buffer_type
            ),
            "buffer_layout": b.buffer_layout,
            "size": b.max_size_per_bank,
            "rank": b.rank,
        }
        buffer_data.append(buffer_dict)

    return {
        "id": operation.operation_id,
        "name": operation.name,
        "buffers": buffer_data,
    }


def serialize_devices(devices):
    return [d.to_dict() for d in devices]


def serialize_operations_buffers(operations, buffers):
    # Pre-serialize all buffers once using optimized method with defaultdict
    serialized_buffers = defaultdict(list)
    for b in buffers:
        buffer_dict = {
            "device_id": b.device_id,
            "address": b.address,
            "buffer_type": (
                b.buffer_type.value
                if hasattr(b.buffer_type, "value")
                else b.buffer_type
            ),
            "buffer_layout": b.buffer_layout,
            "size": b.max_size_per_bank,
            "rank": b.rank,
        }
        serialized_buffers[b.operation_id].append(buffer_dict)

    results = []
    for operation in operations:
        operation_buffers = serialized_buffers[operation.operation_id]
        results.append(
            {
                "id": operation.operation_id,
                "name": operation.name,
                "buffers": operation_buffers,
            }
        )

    return results


def serialize_buffer(buffer):
    return {
        "buffer_type": buffer.buffer_type,
        "device_id": buffer.device_id,
        "size": buffer.max_size_per_bank,
        "address": buffer.address,
        "rank": buffer.rank,
    }


def serialize_tensors(
    tensors, producers_consumers, local_comparisons, global_comparisons
):
    producers_consumers_dict = {
        (pc.tensor_id, pc.rank): pc for pc in producers_consumers
    }
    results = []
    comparisons = comparisons_by_tensor_id(local_comparisons, global_comparisons)
    for tensor in tensors:
        tensor_data = tensor.to_dict()
        tensor_id = tensor_data.pop("tensor_id")
        pc = producers_consumers_dict.get((tensor_id, tensor.rank))
        if pc is None:
            pc = producers_consumers_dict.get((tensor_id, 0))
        tensor_data.update(
            {
                "id": tensor_id,
                "comparison": comparisons.get(tensor_id),
                "consumers": pc.consumers if pc else [],
                "producers": pc.producers if pc else [],
            }
        )
        results.append(tensor_data)

    return results
