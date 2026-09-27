// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { describe, expect, it } from 'vitest';
import { BoundType, ColumnDefinition, ColumnKeys, Columns } from '../src/definitions/PerfTable';
import { OpType } from '../src/definitions/Performance';
import { PerfHeuristicFlag } from '../src/definitions/PerfHeuristics';
import { formatPercentage, formatSize } from '../src/functions/math';
import { formatCellText } from '../src/functions/perfFunctions';
import {
    PerfTableSizingInput,
    getGroupedNumberLength,
    getLongestUnbreakableRunLength,
    getPerfTableSizing,
} from '../src/hooks/usePerfTableColumnSizing';
import { BufferType } from '../src/model/BufferType';
import { TypedPerfTableRow } from '../src/model/PerfTable';

const getColumn = (key: ColumnKeys): ColumnDefinition => {
    const column = Columns.find((candidate) => candidate.key === key);

    if (!column) {
        throw new Error(`No column ${key}`);
    }

    return column;
};

const L1_COLUMN: ColumnDefinition = { name: 'L1 Usage %', key: ColumnKeys.L1Fullness, unit: '%', decimals: 1 };
const OP_COLUMN: ColumnDefinition = { name: 'OP', key: ColumnKeys.OP };
const NPE_COLUMN: ColumnDefinition = { name: 'NPE', key: ColumnKeys.GlobalCallCount };
const HIGH_DISPATCH_COLUMN: ColumnDefinition = { name: 'Slow', key: ColumnKeys.HighDispatch };

const makeRow = (overrides: Partial<TypedPerfTableRow> = {}): TypedPerfTableRow =>
    ({
        id: 1,
        op_type: OpType.DEVICE_OP,
        op_code: 'Matmul',
        raw_op_code: 'Matmul',
        advice: [],
        bound: null,
        isFirstHashOccurrence: true,
        ...overrides,
    }) as unknown as TypedPerfTableRow;

const sizingFor = (overrides: Partial<PerfTableSizingInput>) =>
    getPerfTableSizing({
        rows: [],
        comparisonRows: [],
        visibleColumns: [],
        hasOperations: false,
        npeFileByGlobalCallCount: new Map(),
        isPrimaryActiveReport: true,
        activeReportComparisonIndex: null,
        provideMatmulAdvice: false,
        ...overrides,
    });

describe('getGroupedNumberLength', () => {
    // Choosing the widest row must not need Intl per row, so the shortcut has to agree with it.
    const roundingEdges = [9.96, 99.95, 999.96, 999999.5, 2.5, 0.05, 0.0004];
    const groupingEdges = [0, 1, 10, 999.9, 1000, 1234.5678, 12345.6, 1234567.891, 100000, 7.1e15];
    const fractionEdges = [0.001, 3.14159, 1000000000000.25];
    const negatives = [-1, -0.004, -1234.56];
    const values = [...roundingEdges, ...groupingEdges, ...fractionEdges, ...negatives];

    it.each([0, 1, 2, 3])('matches formatSize with %i fraction digits', (fractionDigits) => {
        values.forEach((value) => {
            expect(getGroupedNumberLength(value, fractionDigits)).toBe(formatSize(value, fractionDigits).length);
        });
    });

    it('matches formatPercentage for the percent columns, less the sign', () => {
        values
            .filter((value) => value >= 0 && value < 1e6)
            .forEach((value) => {
                expect((getGroupedNumberLength(value, 1) ?? 0) + 1).toBe(formatPercentage(value, 1).length);
            });
    });

    it('declines values only Intl can format', () => {
        expect(getGroupedNumberLength(Infinity, 0)).toBeNull();
        expect(getGroupedNumberLength(1e22, 0)).toBeNull();
    });
});

describe('getLongestUnbreakableRunLength', () => {
    it('breaks at whitespace and after hyphens and slashes', () => {
        expect(getLongestUnbreakableRunLength('Reshape MISSING')).toBe('MISSING'.length);
        expect(getLongestUnbreakableRunLength('12.5 GB/s')).toBe('12.5'.length);
        expect(getLongestUnbreakableRunLength('abc-defgh')).toBe('defgh'.length);
        expect(getLongestUnbreakableRunLength('')).toBe(0);
    });
});

describe('getPerfTableSizing', () => {
    it('keeps the cell text with the most characters and the one with the longest unbreakable run', () => {
        const opCode = getColumn(ColumnKeys.OpCode);
        const sizing = sizingFor({
            visibleColumns: [opCode],
            rows: [
                makeRow({ op_code: 'a b c d e f g h i j k' }),
                makeRow({ op_code: 'LongestSingleWord' }),
                makeRow({ op_code: 'short' }),
            ],
        });

        expect(sizing.columns[0].column).toBe(opCode);
        expect(sizing.columns[0].widestText).toBe('a b c d e f g h i j k');
        expect(sizing.columns[0].longestRunText).toBe('LongestSingleWord');
    });

    it('formats the widest number exactly as the table renders it', () => {
        const deviceTime = getColumn(ColumnKeys.DeviceTime);
        const dram = getColumn(ColumnKeys.Dram);
        const rows = [
            makeRow({ device_time: 12.4, dram: 1.25 }),
            makeRow({ device_time: 123456.7, dram: 999.96 }),
            makeRow({ device_time: 99, dram: 12.3 }),
        ];
        const sizing = sizingFor({ visibleColumns: [deviceTime, dram], rows });

        expect(sizing.columns[0].widestText).toBe(formatCellText(123456.7, deviceTime));
        expect(sizing.columns[0].widestText).toBe('123,457 µs');
        expect(sizing.columns[1].widestText).toBe(formatCellText(999.96, dram));
    });

    it('ignores cells the table leaves blank', () => {
        const deviceTime = getColumn(ColumnKeys.DeviceTime);
        const sizing = sizingFor({
            visibleColumns: [deviceTime],
            rows: [
                makeRow({ device_time: 5 }),
                makeRow({ device_time: 123456789, bound: BoundType.HOST }),
                makeRow({ device_time: 987654321, op_type: OpType.SIGNPOST }),
                makeRow({ device_time: null }),
            ],
        });

        expect(sizing.columns[0].widestText).toBe(formatCellText(5, deviceTime));
    });

    it('sizes L1 from every row of an op run, since the table shows it on whichever sorts first', () => {
        const sizing = sizingFor({
            visibleColumns: [L1_COLUMN],
            rows: [makeRow({ op: 1, l1_fullness_percent: 5 }), makeRow({ op: 1, l1_fullness_percent: 55.5 })],
        });

        expect(sizing.columns[0].widestText).toBe(formatPercentage(55.5, 1));
        expect(sizing.columns[0].isNoWrap).toBe(true);
    });

    it('counts comparison rows only for the columns they render', () => {
        const deviceTime = getColumn(ColumnKeys.DeviceTime);
        const id = getColumn(ColumnKeys.Id);
        const sizing = sizingFor({
            visibleColumns: [id, deviceTime],
            rows: [makeRow({ id: 1, device_time: 1 })],
            comparisonRows: [[makeRow({ id: 123456789, device_time: 4567 })]],
        });

        expect(sizing.columns[0].widestText).toBe('1');
        expect(sizing.columns[1].widestText).toBe(formatCellText(4567, deviceTime));
    });

    it('renders the OP column as the raw id once operations have loaded, and grouped before', () => {
        const row = makeRow({ op: 12345 });

        expect(sizingFor({ visibleColumns: [OP_COLUMN], rows: [row], hasOperations: true }).columns[0].widestText).toBe(
            '12345',
        );
        expect(sizingFor({ visibleColumns: [OP_COLUMN], rows: [row] }).columns[0].widestText).toBe('12,345');
    });

    it('marks the NPE link only on primary rows whose call count is in the manifest', () => {
        const npeFileByGlobalCallCount = new Map([[2049, 'npe.zst']]);

        expect(
            sizingFor({
                visibleColumns: [NPE_COLUMN],
                rows: [makeRow({ global_call_count: 2049 })],
                npeFileByGlobalCallCount,
            }).columns[0],
        ).toMatchObject({ hasNpeLink: true, widestText: '' });

        // Comparison sub-rows print the call count as text instead of linking
        expect(
            sizingFor({
                visibleColumns: [NPE_COLUMN],
                rows: [makeRow({ global_call_count: 7 })],
                comparisonRows: [[makeRow({ global_call_count: 2049 })]],
                npeFileByGlobalCallCount,
            }).columns[0],
        ).toMatchObject({ hasNpeLink: false, widestText: '2,049' });
    });

    it('collects icons and the widest memory tag', () => {
        const sizing = sizingFor({
            visibleColumns: [HIGH_DISPATCH_COLUMN, getColumn(ColumnKeys.BufferType)],
            rows: [
                makeRow({ device_time: 100, buffer_type: BufferType.L1 }),
                makeRow({ device_time: 1, buffer_type: BufferType.SYSTEM_MEMORY }),
            ],
        });

        expect(sizing.columns[0].hasIcon).toBe(true);
        expect(sizing.columns[1].widestMemoryLabel).toBe('System Memory');
    });

    it('keeps the flags of each row as a separate set rather than merging them', () => {
        const dramOnly = [PerfHeuristicFlag.DRAM_BOUND];
        const lowUtilOnly = [PerfHeuristicFlag.LOW_UTILISATION];
        const dramAndRecompute = [PerfHeuristicFlag.DRAM_BOUND, PerfHeuristicFlag.RECOMPUTE_CANDIDATE];

        const { flagSets } = sizingFor({
            visibleColumns: [getColumn(ColumnKeys.Flags)],
            rows: [
                makeRow({ heuristicFlags: dramOnly }),
                makeRow({ heuristicFlags: lowUtilOnly }),
                makeRow({ heuristicFlags: [...dramOnly] }),
                makeRow({ heuristicFlags: dramAndRecompute }),
                makeRow({ heuristicFlags: [] }),
                makeRow(),
            ],
        }).columns[0];

        // No row carries both DRAM bound and Low util, so no sample may put them side by side
        expect(flagSets).toEqual([dramOnly, lowUtilOnly, dramAndRecompute]);
    });

    it('reports a tensor trigger only when the active report has a non-signpost row', () => {
        const signpost = makeRow({ op_type: OpType.SIGNPOST });

        expect(sizingFor({ rows: [signpost] }).hasTensorTrigger).toBe(false);
        expect(sizingFor({ rows: [signpost, makeRow()] }).hasTensorTrigger).toBe(true);
        expect(
            sizingFor({
                rows: [makeRow()],
                comparisonRows: [[signpost]],
                isPrimaryActiveReport: false,
                activeReportComparisonIndex: 0,
            }).hasTensorTrigger,
        ).toBe(false);
    });

    it('sizes matmul advice only while advice is shown', () => {
        const rows = [makeRow({ advice: ['Short tip', 'A considerably longer piece of matmul advice'] })];

        expect(sizingFor({ rows }).widestAdvice).toBe('');
        expect(sizingFor({ rows, provideMatmulAdvice: true })).toMatchObject({
            widestAdvice: 'A considerably longer piece of matmul advice',
            longestRunAdvice: 'A considerably longer piece of matmul advice',
        });
    });
});

/** Rows whose `key` counts its reads, so a case can tell which columns were measured again. */
const rowsCountingReadsOf = (key: keyof TypedPerfTableRow, rows: TypedPerfTableRow[]) => {
    const reads = { count: 0 };

    for (const row of rows) {
        const value = row[key];

        Object.defineProperty(row, key, {
            enumerable: true,
            get: () => {
                reads.count += 1;
                return value;
            },
        });
    }

    return { rows, reads };
};

// Toggling a column (high dispatch, the column picker) used to measure every column over every
// row again. Each case counts reads of one column's field to see whether that column was re-measured.
describe('getPerfTableSizing across calls', () => {
    const mathFidelity = getColumn(ColumnKeys.MathFidelity);
    const deviceTime = getColumn(ColumnKeys.DeviceTime);

    it('measures a column once for a set of rows, however the other columns change', () => {
        const { rows, reads } = rowsCountingReadsOf('math_fidelity', [
            makeRow({ math_fidelity: 'HiFi4' }),
            makeRow({ math_fidelity: 'LoFi' }),
        ]);

        const first = sizingFor({ rows, visibleColumns: [mathFidelity] });
        const readsForOneMeasurement = reads.count;
        const withColumnAdded = sizingFor({ rows, visibleColumns: [mathFidelity, HIGH_DISPATCH_COLUMN] });
        const withColumnRemoved = sizingFor({ rows, visibleColumns: [mathFidelity] });

        expect(reads.count).toBe(readsForOneMeasurement);
        expect(withColumnAdded.columns.map(({ column }) => column)).toEqual([mathFidelity, HIGH_DISPATCH_COLUMN]);
        expect(withColumnAdded.columns[0]).toEqual(first.columns[0]);
        expect(withColumnRemoved).toEqual(first);
    });

    it('measures again for a new set of rows', () => {
        const { rows, reads } = rowsCountingReadsOf('math_fidelity', [makeRow({ math_fidelity: 'HiFi4' })]);

        sizingFor({ rows, visibleColumns: [mathFidelity] });
        const readsForOneMeasurement = reads.count;
        sizingFor({ rows: [...rows], visibleColumns: [mathFidelity] });

        expect(reads.count).toBe(readsForOneMeasurement * 2);
    });

    it('re-measures the OP column alone once operations load', () => {
        const { rows, reads } = rowsCountingReadsOf('op', [makeRow({ op: 12345, math_fidelity: 'HiFi4' })]);

        expect(sizingFor({ rows, visibleColumns: [OP_COLUMN, mathFidelity] }).columns[0].widestText).toBe('12,345');
        const readsBeforeLoad = reads.count;

        expect(
            sizingFor({ rows, visibleColumns: [OP_COLUMN, mathFidelity], hasOperations: true }).columns[0].widestText,
        ).toBe('12345');
        expect(reads.count).toBeGreaterThan(readsBeforeLoad);
    });

    it('re-measures the NPE column when the manifest changes', () => {
        const rows = [makeRow({ global_call_count: 2049 })];

        expect(sizingFor({ rows, visibleColumns: [NPE_COLUMN] }).columns[0].hasNpeLink).toBe(false);
        expect(
            sizingFor({
                rows,
                visibleColumns: [NPE_COLUMN],
                npeFileByGlobalCallCount: new Map([[2049, 'npe.zst']]),
            }).columns[0].hasNpeLink,
        ).toBe(true);
    });

    // The comparison tab hands its datasets over in a list it rebuilds on every render.
    it('re-measures comparison columns for new comparison rows, not for a rebuilt list of the same ones', () => {
        const rows = [makeRow({ device_time: 1 })];
        const { rows: comparisonDataset, reads } = rowsCountingReadsOf('device_time', [makeRow({ device_time: 4567 })]);

        sizingFor({ rows, visibleColumns: [deviceTime], comparisonRows: [comparisonDataset] });
        const readsForOneMeasurement = reads.count;
        sizingFor({ rows, visibleColumns: [deviceTime], comparisonRows: [comparisonDataset] });

        expect(reads.count).toBe(readsForOneMeasurement);

        const widerDataset = [makeRow({ device_time: 1234567 })];

        expect(
            sizingFor({ rows, visibleColumns: [deviceTime], comparisonRows: [widerDataset] }).columns[0].widestText,
        ).toBe(formatCellText(1234567, deviceTime));
    });
});
