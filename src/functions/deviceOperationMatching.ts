// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { PerfTableRow } from '../model/PerfTable';
import { DeviceOperationMapping, DeviceOperationOrderCandidates } from '../model/DeviceOperationMapping';
import { OpType } from '../definitions/Performance';

/**
 * What every failed match returns. One shared value rather than a fresh `[]` per
 * attempt: the match is memoised on its inputs, which settle one query at a time
 * while a report loads, and a new empty identity per attempt would invalidate
 * everything derived from the match — the enriched perf rows among them — for a
 * result that never changed. Frozen because every consumer shares it.
 */
export const NO_DEVICE_OPERATION_MATCHES: readonly DeviceOperationMapping[] = Object.freeze([]);

/**
 * TODO: remove once memory and performance reports carry a shared run id (#1800)
 * @description Drop per-device duplicates of the same device operation, keeping
 * only keys seen exactly once per device. Some multi-device memory reports
 * record each device op once per device and others record it once, and nothing
 * in the report distinguishes the two shapes — so this is a fallback for the
 * duplicated shape rather than an unconditional normalisation. See #1810.
 */
export const collapseMultideviceOperations = (
    deviceOperations: DeviceOperationMapping[],
    numDevices: number,
): DeviceOperationMapping[] => {
    // A single device has no per-device duplicates, and an unknown device count
    // (0, before the devices query settles) gives no count to collapse on.
    if (numDevices <= 1) {
        return deviceOperations;
    }

    const operationCountByKey = new Map<string, number>();

    for (const { name, id } of deviceOperations) {
        const key = `${name}-${id}`;
        operationCountByKey.set(key, (operationCountByKey.get(key) || 0) + 1);
    }

    const collapsed: DeviceOperationMapping[] = [];
    const seen = new Set<string>();

    for (const deviceOperation of deviceOperations) {
        const key = `${deviceOperation.name}-${deviceOperation.id}`;

        if (!seen.has(key) && operationCountByKey.get(key) === numDevices) {
            collapsed.push(deviceOperation);
            seen.add(key);
        }
    }

    return collapsed;
};

/**
 * @description Pair each device operation with the perf row at the same index,
 * or return `NO_DEVICE_OPERATION_MATCHES` if any position disagrees. Trailing
 * perf rows are tolerated so a report that also lists host ops still matches.
 * Validation runs before any allocation because callers try more than one
 * candidate list, and the mappings are copies so a rejected attempt leaves the
 * caller's list untouched.
 */
const alignToPerfRows = (
    deviceOperations: DeviceOperationMapping[],
    perfRows: PerfTableRow[],
): readonly DeviceOperationMapping[] => {
    if (deviceOperations.length === 0 || deviceOperations.length > perfRows.length) {
        return NO_DEVICE_OPERATION_MATCHES;
    }

    const isAligned = deviceOperations.every(
        (deviceOperation, index) => perfRows[index].raw_op_code === deviceOperation.name,
    );

    if (!isAligned) {
        return NO_DEVICE_OPERATION_MATCHES;
    }

    return deviceOperations.map((deviceOperation, index) => ({ ...deviceOperation, perfData: perfRows[index] }));
};

/**
 * @description Drop the signpost rows a model emits with `signpost()`. They are
 * markers rather than operations, so the memory report has no device operation
 * to pair them with, and they are their own op type rather than a host op — so
 * the pinned `hideHostOps` filter leaves them in place. Alignment is positional,
 * so one signpost anywhere but the tail shifts every later row out of position
 * and fails the whole report. See #1943.
 */
const alignableRowsOf = (perfRows: PerfTableRow[]): PerfTableRow[] =>
    perfRows.filter((perfRow) => perfRow.op_type !== OpType.SIGNPOST);

const alignCollapsedToPerfRows = (
    deviceOperations: DeviceOperationMapping[],
    perfRows: PerfTableRow[],
    numDevices: number,
): readonly DeviceOperationMapping[] => {
    if (numDevices <= 1) {
        return NO_DEVICE_OPERATION_MATCHES;
    }

    const collapsedOperations = collapseMultideviceOperations(deviceOperations, numDevices);

    // A valid per-device list contributes every operation exactly once per
    // device. A smaller subset can prefix-match and falsely mark a report linked.
    if (collapsedOperations.length * numDevices !== deviceOperations.length) {
        return NO_DEVICE_OPERATION_MATCHES;
    }

    return alignToPerfRows(collapsedOperations, perfRows);
};

/**
 * @description Match a memory report's device operations against a performance
 * report's rows, returning the mappings when the two sequences describe the same
 * run and `NO_DEVICE_OPERATION_MATCHES` when they don't (the report-link UNLINKED
 * signal).
 *
 * The raw list is tried first because it is the shape most reports have; only
 * when that fails do we assume the memory report duplicated each op per device
 * and retry against the collapsed list.
 */
export const matchDeviceOperationsToPerf = (
    deviceOperations: DeviceOperationMapping[],
    perfRows: PerfTableRow[],
    numDevices: number,
): readonly DeviceOperationMapping[] => {
    const alignableRows = alignableRowsOf(perfRows);
    const directMatch = alignToPerfRows(deviceOperations, alignableRows);

    if (directMatch.length > 0) {
        return directMatch;
    }

    return alignCollapsedToPerfRows(deviceOperations, alignableRows, numDevices);
};

const hasSameDeviceOperations = (
    functionStartOperations: DeviceOperationMapping[],
    functionEndOperations: DeviceOperationMapping[],
): boolean => {
    if (functionStartOperations.length !== functionEndOperations.length) {
        return false;
    }

    const operationCountByKey = new Map<string, number>();

    for (const { id, name } of functionStartOperations) {
        const key = JSON.stringify([id, name]);
        operationCountByKey.set(key, (operationCountByKey.get(key) ?? 0) + 1);
    }

    for (const { id, name } of functionEndOperations) {
        const key = JSON.stringify([id, name]);
        const remaining = operationCountByKey.get(key);

        if (!remaining) {
            return false;
        }

        operationCountByKey.set(key, remaining - 1);
    }

    return true;
};

/**
 * TODO: remove once memory and performance reports carry a shared run id (#1800)
 * @description Match nested device operations whose profiler events can be
 * child-first because a child's workload is enqueued before its parent's. Start
 * order remains preferred for reports whose profiler rows are parent-first;
 * end order is the fallback when the same operations are child-first. Raw orders
 * are tried before either multi-device collapse so a spurious collapsed prefix
 * cannot suppress the complete end-order match. See #1860.
 */
export const matchDeviceOperationOrdersToPerf = (
    { functionStartOperations, functionEndOperations }: DeviceOperationOrderCandidates,
    perfRows: PerfTableRow[],
    numDevices: number,
): readonly DeviceOperationMapping[] => {
    const alignableRows = alignableRowsOf(perfRows);
    const functionStartMatch = alignToPerfRows(functionStartOperations, alignableRows);

    if (functionStartMatch.length > 0) {
        return functionStartMatch;
    }

    // An interrupted capture can omit function-end events. Since alignment
    // tolerates trailing perf rows, only a complete reordering is safe to retry.
    if (!hasSameDeviceOperations(functionStartOperations, functionEndOperations)) {
        return NO_DEVICE_OPERATION_MATCHES;
    }

    const functionEndMatch = alignToPerfRows(functionEndOperations, alignableRows);

    if (functionEndMatch.length > 0) {
        return functionEndMatch;
    }

    const collapsedFunctionStartMatch = alignCollapsedToPerfRows(functionStartOperations, alignableRows, numDevices);

    if (collapsedFunctionStartMatch.length > 0) {
        return collapsedFunctionStartMatch;
    }

    return alignCollapsedToPerfRows(functionEndOperations, alignableRows, numDevices);
};
