// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { PerfTableRow } from '../model/PerfTable';

const clampToRange = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

/**
 * @description The op<->perf mapping is resolved against the link-pinned report
 * while the performance slider spans the rows the performance tab is currently
 * displaying, so a mapped perf id can fall outside a narrowed view (a signpost
 * window, most obviously). Keep the handles inside their own track (#1812).
 *
 * A selection that misses the track entirely widens to the whole track rather
 * than clamping, because clamping would pin both handles to the same edge and
 * that reads as a legitimate one-row selection. `Performance.tsx` derives the
 * table's range through this same helper, so the widen branch is also its
 * fall back to the full report — one definition of "misses the track" for both,
 * rather than two that have to be kept in step by hand.
 */
export const clampSelectionToRange = (
    min: number,
    max: number,
    rangeMin: number,
    rangeMax: number,
): [number, number] => {
    const missesRange = max < rangeMin || min > rangeMax;

    return missesRange
        ? [rangeMin, rangeMax]
        : [clampToRange(min, rangeMin, rangeMax), clampToRange(max, rangeMin, rangeMax)];
};

/**
 * @description The lowest and highest row id in the report, or `null` when it has
 * none. One pass rather than spreading the ids into `Math.min`/`Math.max`: an
 * engine caps how many arguments a call can take, so a large enough report throws
 * a `RangeError` there. A row whose id does not parse is skipped rather than
 * turning both ends into `NaN`.
 */
export const getPerformanceRange = (rows: PerfTableRow[] | undefined): [number, number] | null => {
    let min = Infinity;
    let max = -Infinity;

    for (const row of rows ?? []) {
        const id = parseInt(row.id, 10);

        if (id < min) {
            min = id;
        }

        if (id > max) {
            max = id;
        }
    }

    return min <= max ? [min, max] : null;
};
