// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { useMemo } from 'react';
import { ColumnDefinition, ColumnKeys, comparisonKeys } from '../definitions/PerfTable';
import { HIGH_DISPATCH_THRESHOLD_US, OpType } from '../definitions/Performance';
import { PerfHeuristicFlag } from '../definitions/PerfHeuristics';
import { BufferType, BufferTypeLabel } from '../model/BufferType';
import { TypedPerfTableRow } from '../model/PerfTable';
import { PerfCellValue, formatCellText, hasMatmulAdvice, isSuppressedCell } from '../functions/perfFunctions';

/**
 * The widest content any row puts in one column, as samples the sizer row renders for real so the
 * browser's own table layout reads them. Text is kept as the real cell string rather than a length so
 * wrapping, glyph widths and break opportunities are the browser's, not an estimate.
 */
export interface PerfTableColumnSizing {
    column: ColumnDefinition;
    /** Cell text with the most characters: the column's max-content width. */
    widestText: string;
    /** Cell text holding the longest unbreakable run: the column's min-content width. */
    longestRunText: string;
    isNoWrap: boolean;
    hasIcon: boolean;
    widestMemoryLabel: string;
    /**
     * Every distinct set of flags a row carries. A cell holds one row's flags, so each set is its own
     * sample: sized together, flags no single row combines would widen the column. Rows list their flags
     * in a fixed order, so four flags make at most fifteen sets.
     */
    flagSets: PerfHeuristicFlag[][];
    hasNpeLink: boolean;
}

export interface PerfTableSizing {
    hasTensorTrigger: boolean;
    columns: PerfTableColumnSizing[];
    widestAdvice: string;
    longestRunAdvice: string;
}

export interface PerfTableSizingInput {
    rows: TypedPerfTableRow[];
    comparisonRows: TypedPerfTableRow[][];
    visibleColumns: ColumnDefinition[];
    hasOperations: boolean;
    npeFileByGlobalCallCount: ReadonlyMap<number, string>;
    isPrimaryActiveReport: boolean;
    activeReportComparisonIndex: number | null;
    provideMatmulAdvice: boolean;
}

// Intl's default `maximumFractionDigits` for decimal style, which `formatSize` inherits when a column
// sets no `decimals`; `formatPercentage` defaults to 2 itself.
const INTL_DEFAULT_FRACTION_DIGITS = 3;
const PERCENT_DEFAULT_FRACTION_DIGITS = 2;
// Past this Number#toString switches to exponent notation while Intl keeps every digit.
const PLAIN_INTEGER_LIMIT = 1e21;
const DIGIT_GROUP_SIZE = 3;

const isBreakingWhitespace = (charCode: number) =>
    charCode === 32 || charCode === 9 || charCode === 10 || charCode === 12 || charCode === 13;

// Browsers may wrap after a hyphen or slash as well as at spaces.
const isBreakAfter = (charCode: number) => charCode === 45 || charCode === 47;

/** Length of the longest run of text a line cannot break inside, approximating the browser's rules. */
export const getLongestUnbreakableRunLength = (text: string): number => {
    let longest = 0;
    let current = 0;

    for (let index = 0; index < text.length; index++) {
        const charCode = text.charCodeAt(index);

        if (isBreakingWhitespace(charCode)) {
            current = 0;
        } else {
            current++;
            longest = Math.max(longest, current);

            if (isBreakAfter(charCode)) {
                current = 0;
            }
        }
    }

    return longest;
};

/**
 * Character count of an en-US number rounded to `maximumFractionDigits` with digit grouping, matching
 * `formatSize` without building an `Intl.NumberFormat` per row. `null` when only Intl can say.
 */
export const getGroupedNumberLength = (value: number, maximumFractionDigits: number): number | null => {
    const magnitude = Math.abs(value);
    const scale = 10 ** maximumFractionDigits;
    let integerPart: number;
    let fractionLength = 0;

    if (!Number.isFinite(magnitude) || magnitude >= PLAIN_INTEGER_LIMIT) {
        return null;
    }

    if (Number.isInteger(magnitude)) {
        integerPart = magnitude;
    } else {
        const scaled = Math.round(magnitude * scale);

        if (!Number.isSafeInteger(scaled)) {
            return null;
        }

        integerPart = Math.floor(scaled / scale);
        let fraction = scaled - integerPart * scale;
        fractionLength = fraction === 0 ? 0 : maximumFractionDigits;

        while (fraction !== 0 && fraction % 10 === 0) {
            fraction /= 10;
            fractionLength--;
        }
    }

    let integerLength = 1;

    for (let nextPowerOfTen = 10; integerPart >= nextPowerOfTen; nextPowerOfTen *= 10) {
        integerLength++;
    }

    const groupSeparatorCount = Math.floor((integerLength - 1) / DIGIT_GROUP_SIZE);

    return (value < 0 ? 1 : 0) + integerLength + groupSeparatorCount + (fractionLength > 0 ? fractionLength + 1 : 0);
};

interface TextCandidate {
    length: number;
    // A numeric cell defers formatting until it wins, so Intl runs once per column rather than per row
    row: TypedPerfTableRow | null;
    text: string;
}

interface ColumnAccumulator {
    column: ColumnDefinition;
    isNoWrap: boolean;
    unitRunLength: number;
    widest: TextCandidate;
    longestRun: TextCandidate;
    hasIcon: boolean;
    widestMemoryLabel: string;
    flagSets: Map<string, PerfHeuristicFlag[]>;
    hasNpeLink: boolean;
}

const createCandidate = (): TextCandidate => ({ length: 0, row: null, text: '' });

export const getFlagSetKey = (flags: PerfHeuristicFlag[]) => flags.join(',');

const createAccumulator = (column: ColumnDefinition): ColumnAccumulator => ({
    column,
    isNoWrap: column.key === ColumnKeys.L1Fullness,
    unitRunLength: column.unit && column.unit !== '%' ? getLongestUnbreakableRunLength(column.unit) : 0,
    widest: createCandidate(),
    longestRun: createCandidate(),
    hasIcon: false,
    widestMemoryLabel: '',
    flagSets: new Map<string, PerfHeuristicFlag[]>(),
    hasNpeLink: false,
});

const considerText = (accumulator: ColumnAccumulator, text: string) => {
    if (text.length > accumulator.widest.length) {
        accumulator.widest = { length: text.length, row: null, text };
    }

    // No run is longer than its text, so most rows are settled without scanning them
    if (text.length <= accumulator.longestRun.length) {
        return;
    }

    const runLength = getLongestUnbreakableRunLength(text);

    if (runLength > accumulator.longestRun.length) {
        accumulator.longestRun = { length: runLength, row: null, text };
    }
};

const considerFormattedNumber = (accumulator: ColumnAccumulator, row: TypedPerfTableRow, value: number) => {
    const { key, unit, decimals } = accumulator.column;
    const isPercent = unit === '%';
    const numberLength =
        key === ColumnKeys.Id
            ? null
            : getGroupedNumberLength(
                  value,
                  decimals ?? (isPercent ? PERCENT_DEFAULT_FRACTION_DIGITS : INTL_DEFAULT_FRACTION_DIGITS),
              );

    if (numberLength === null) {
        considerText(accumulator, formatCellText(value, accumulator.column));
        return;
    }

    // The unit is identical on every row, so the number alone decides which row is widest. Percentages
    // take a bare '%', every other unit a space and its name.
    let unitLength = 0;

    if (isPercent) {
        unitLength = 1;
    } else if (unit) {
        unitLength = unit.length + 1;
    }

    const length = numberLength + unitLength;
    const runLength = isPercent ? length : Math.max(numberLength, accumulator.unitRunLength);

    if (length > accumulator.widest.length) {
        accumulator.widest = { length, row, text: '' };
    }

    if (runLength > accumulator.longestRun.length) {
        accumulator.longestRun = { length: runLength, row, text: '' };
    }
};

// Mirrors the branches of `formatCell` (and PerfTableRowGroup's `PerfNpeLink`) that decide what a cell renders.
const considerCell = (
    accumulator: ColumnAccumulator,
    row: TypedPerfTableRow,
    isPrimaryRow: boolean,
    { hasOperations, npeFileByGlobalCallCount }: PerfTableSizingInput,
) => {
    const { key } = accumulator.column;
    const value: PerfCellValue = row[key];

    if (key === ColumnKeys.GlobalCallCount && isPrimaryRow) {
        // NPE uses 0 as its "no value", so it is mapped out of the lookup rather than matched
        if (npeFileByGlobalCallCount.has(parseInt(String(value), 10) || -1)) {
            accumulator.hasNpeLink = true;
        }
        return;
    }

    if (key === ColumnKeys.Flags) {
        const flags = row.heuristicFlags;

        if (flags && flags.length > 0) {
            accumulator.flagSets.set(getFlagSetKey(flags), flags);
        }
        return;
    }

    // Only the first row of an op run shows its L1 value, but every row of the run carries the same
    // snapshot, so sizing as if each were first is exact and independent of the sort order.
    if (isSuppressedCell(row, key)) {
        return;
    }

    if (key === ColumnKeys.BufferType) {
        const label = BufferTypeLabel[value as BufferType] ?? '';

        if (label.length > accumulator.widestMemoryLabel.length) {
            accumulator.widestMemoryLabel = label;
        }
    } else if (key === ColumnKeys.HighDispatch) {
        if (row.device_time !== null && row.device_time > HIGH_DISPATCH_THRESHOLD_US) {
            accumulator.hasIcon = true;
        }
    } else if (key === ColumnKeys.CacheHit) {
        if (typeof value === 'boolean' && !row.isFirstHashOccurrence) {
            accumulator.hasIcon = true;
        }
    } else if (key === ColumnKeys.OP && hasOperations) {
        considerText(accumulator, typeof value === 'number' || typeof value === 'string' ? String(value) : '');
    } else if (key === ColumnKeys.Hash) {
        considerText(accumulator, value?.toString() || '');
    } else if (typeof value === 'number') {
        considerFormattedNumber(accumulator, row, value);
    } else if (key !== ColumnKeys.L1Fullness) {
        considerText(accumulator, formatCellText(value, accumulator.column));
    }
};

const resolveCandidate = (candidate: TextCandidate, column: ColumnDefinition) =>
    candidate.row ? formatCellText(candidate.row[column.key], column) : candidate.text;

const isTensorTriggerRow = (row: TypedPerfTableRow | undefined) => row !== undefined && row.op_type !== OpType.SIGNPOST;

type ColumnContent = Omit<PerfTableColumnSizing, 'column'>;

/** A column's content, with every input it was measured from that can differ between calls on the same rows. */
interface MeasuredColumn {
    unit: ColumnDefinition['unit'];
    decimals: ColumnDefinition['decimals'];
    comparisonRows: TypedPerfTableRow[][];
    hasOperations: boolean;
    npeFileByGlobalCallCount: ReadonlyMap<number, string>;
    content: ColumnContent;
}

// Adding or removing one column (the high-dispatch toggle, the column picker) changes nothing about
// the others, so their measurements are kept rather than taken again over every row. Keyed weakly on
// the rows, so a measurement lives exactly as long as the rows it was taken from and a new report
// needs no reset.
const measuredColumnsByRows = new WeakMap<TypedPerfTableRow[], Map<ColumnKeys, MeasuredColumn>>();

const isSameDatasets = (a: TypedPerfTableRow[][], b: TypedPerfTableRow[][]) =>
    a.length === b.length && a.every((dataset, index) => dataset === b[index]);

// Only the inputs a column actually reads invalidate it: the NPE manifest arriving, say, re-measures
// the NPE column alone. The comparison datasets are compared one by one because the comparison tab
// hands them over in a list built afresh on every render.
const isMeasurementCurrent = (measured: MeasuredColumn, column: ColumnDefinition, input: PerfTableSizingInput) =>
    measured.unit === column.unit &&
    measured.decimals === column.decimals &&
    (column.key !== ColumnKeys.OP || measured.hasOperations === input.hasOperations) &&
    (column.key !== ColumnKeys.GlobalCallCount ||
        measured.npeFileByGlobalCallCount === input.npeFileByGlobalCallCount) &&
    (!comparisonKeys.includes(column.key) || isSameDatasets(measured.comparisonRows, input.comparisonRows));

const toColumnContent = (accumulator: ColumnAccumulator): ColumnContent => ({
    widestText: resolveCandidate(accumulator.widest, accumulator.column),
    longestRunText: resolveCandidate(accumulator.longestRun, accumulator.column),
    isNoWrap: accumulator.isNoWrap,
    hasIcon: accumulator.hasIcon,
    widestMemoryLabel: accumulator.widestMemoryLabel,
    flagSets: [...accumulator.flagSets.values()],
    hasNpeLink: accumulator.hasNpeLink,
});

// One pass over the rows for all the columns together rather than one per column: a report's rows
// run to tens of MB, and reading them once is markedly faster than reading them once per column.
const measureColumns = (columns: ColumnDefinition[], input: PerfTableSizingInput): ColumnContent[] => {
    if (columns.length === 0) {
        return [];
    }

    const accumulators = columns.map(createAccumulator);
    const comparisonAccumulators = accumulators.filter(({ column }) => comparisonKeys.includes(column.key));

    for (const row of input.rows) {
        for (const accumulator of accumulators) {
            considerCell(accumulator, row, true, input);
        }
    }

    if (comparisonAccumulators.length > 0) {
        for (const dataset of input.comparisonRows) {
            for (const row of dataset) {
                for (const accumulator of comparisonAccumulators) {
                    considerCell(accumulator, row, false, input);
                }
            }
        }
    }

    return accumulators.map(toColumnContent);
};

const getMeasuredColumns = (rows: TypedPerfTableRow[]): Map<ColumnKeys, MeasuredColumn> => {
    let measuredColumns = measuredColumnsByRows.get(rows);

    if (measuredColumns === undefined) {
        measuredColumns = new Map<ColumnKeys, MeasuredColumn>();
        measuredColumnsByRows.set(rows, measuredColumns);
    }

    return measuredColumns;
};

const getAdviceSizing = (rows: TypedPerfTableRow[]) => {
    const widestAdvice = createCandidate();
    const longestRunAdvice = createCandidate();

    for (const row of rows) {
        if (hasMatmulAdvice(row)) {
            for (const advice of row.advice) {
                if (advice.length > widestAdvice.length) {
                    widestAdvice.length = advice.length;
                    widestAdvice.text = advice;
                }

                const runLength = getLongestUnbreakableRunLength(advice);

                if (runLength > longestRunAdvice.length) {
                    longestRunAdvice.length = runLength;
                    longestRunAdvice.text = advice;
                }
            }
        }
    }

    return { widestAdvice: widestAdvice.text, longestRunAdvice: longestRunAdvice.text };
};

const NO_ADVICE_SIZING = { widestAdvice: '', longestRunAdvice: '' };

/**
 * Collects, over every row rather than only the rendered window, the content that decides each
 * column's width. Comparison sub-rows count only for the columns they render. A column already
 * measured over the same rows is not measured again.
 */
export const getPerfTableSizing = (input: PerfTableSizingInput): PerfTableSizing => {
    const { rows, comparisonRows, visibleColumns, isPrimaryActiveReport, activeReportComparisonIndex } = input;
    const activeRows = isPrimaryActiveReport ? rows : (comparisonRows[activeReportComparisonIndex ?? -1] ?? []);
    const measuredColumns = getMeasuredColumns(rows);
    const currentContent = visibleColumns.map((column) => {
        const measured = measuredColumns.get(column.key);

        return measured !== undefined && isMeasurementCurrent(measured, column, input) ? measured.content : null;
    });
    const staleColumns = visibleColumns.filter((_column, index) => currentContent[index] === null);
    const freshContent = measureColumns(staleColumns, input);
    let freshIndex = 0;

    const columns = visibleColumns.map((column, index) => {
        let content = currentContent[index];

        if (content === null) {
            content = freshContent[freshIndex];
            freshIndex += 1;
            measuredColumns.set(column.key, {
                unit: column.unit,
                decimals: column.decimals,
                comparisonRows,
                hasOperations: input.hasOperations,
                npeFileByGlobalCallCount: input.npeFileByGlobalCallCount,
                content,
            });
        }

        return { column, ...content };
    });

    return {
        hasTensorTrigger: activeRows.some(isTensorTriggerRow),
        columns,
        ...(input.provideMatmulAdvice ? getAdviceSizing(rows) : NO_ADVICE_SIZING),
    };
};

/**
 * Sizing for a virtualised perf table. Pass the unsorted rows: sorting reorders them without changing
 * any column's widest content, so a sort click does not rescan the report.
 */
const usePerfTableColumnSizing = ({
    rows,
    comparisonRows,
    visibleColumns,
    hasOperations,
    npeFileByGlobalCallCount,
    isPrimaryActiveReport,
    activeReportComparisonIndex,
    provideMatmulAdvice,
}: PerfTableSizingInput): PerfTableSizing =>
    useMemo(
        () =>
            getPerfTableSizing({
                rows,
                comparisonRows,
                visibleColumns,
                hasOperations,
                npeFileByGlobalCallCount,
                isPrimaryActiveReport,
                activeReportComparisonIndex,
                provideMatmulAdvice,
            }),
        [
            rows,
            comparisonRows,
            visibleColumns,
            hasOperations,
            npeFileByGlobalCallCount,
            isPrimaryActiveReport,
            activeReportComparisonIndex,
            provideMatmulAdvice,
        ],
    );

export default usePerfTableColumnSizing;
