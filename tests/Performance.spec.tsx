// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import '@testing-library/jest-dom/vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { useAtom, useSetAtom } from 'jotai';
import { useState } from 'react';
import { Mock, afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Performance from '../src/routes/Performance';
import {
    useL1PressureByOperation,
    useOpToPerfIdFiltered,
    usePerfFolderList,
    usePerfMeta,
    usePerfMetas,
    usePerformanceComparisonReport,
    usePerformanceRange,
    usePerformanceReport,
} from '../src/hooks/useAPI';
import {
    activePerformanceReportAtom,
    bufferTypeFilterListAtom,
    durationBucketFilterListAtom,
    layoutFilterListAtom,
    mathFilterListAtom,
    perfSelectedTabAtom,
    rawOpCodeFilterListAtom,
    selectedPerfRowIdAtom,
    selectedPerformanceRangeAtom,
} from '../src/store/app';
import { BufferType } from '../src/model/BufferType';
import { DeviceOperationLayoutTypes } from '../src/model/APIData';
import { L1PressureStatus } from '../src/model/L1Pressure';
import { PerfTableRow, TypedPerfTableRow } from '../src/model/PerfTable';
import { PerfHeuristicFlag } from '../src/definitions/PerfHeuristics';
import { OpType, PerfTabIds } from '../src/definitions/Performance';
import { Marker } from '../src/definitions/PerfTable';
import { enrichRowData } from '../src/functions/enrichPerfRowData';
import { TestProviders } from './helpers/TestProviders';

const perfReportProps = vi.hoisted(() => vi.fn());
const chartsTabProps = vi.hoisted(() => vi.fn());

vi.mock('../src/hooks/useAPI.tsx', () => ({
    useL1PressureByOperation: vi.fn(),
    useOpToPerfIdFiltered: vi.fn(),
    usePerfFolderList: vi.fn(),
    usePerfMeta: vi.fn(),
    usePerfMetas: vi.fn(),
    usePerformanceComparisonReport: vi.fn(),
    usePerformanceRange: vi.fn(),
    usePerformanceReport: vi.fn(),
}));

vi.mock('../src/functions/getServerConfig', () => ({
    default: () => ({ SERVER_MODE: true }),
    isFlagEnabled: (value: unknown) => value === true || value === 'true' || value === '1',
}));

// Stub the report shell so these tests exercise only the route's own
// active-report effects, not PerfTable's selection cleanup effect (which would
// otherwise clear selectedPerfRowIdAtom while the skeleton is mounted).
vi.mock('../src/components/performance/PerfReport', () => ({
    default: (props: Record<string, unknown>) => {
        perfReportProps(props);
        return <div data-testid='perf-report-stub' />;
    },
}));

// Stubbed like the report shell: the op-code selection cases read what the route hands the charts.
vi.mock('../src/components/performance/PerformanceChartsTab', () => ({
    default: (props: Record<string, unknown>) => {
        chartsTabProps(props);
        return <div data-testid='charts-tab-stub' />;
    },
}));

// Wrapped rather than replaced, so a case can see which rows the route enriched.
vi.mock('../src/functions/enrichPerfRowData', async (importOriginal) => {
    const actual = await importOriginal<typeof import('../src/functions/enrichPerfRowData')>();

    return { ...actual, enrichRowData: vi.fn(actual.enrichRowData) };
});

const DRAM_PERF_ROW = {
    id: '1',
    global_call_count: 0,
    advice: [],
    total_percent: '12.5',
    bound: 'DRAM',
    op_code: 'Matmul',
    raw_op_code: 'Matmul',
    device: '0',
    device_time: '123.4',
    op_to_op_gap: '2.5',
    cores: '64',
    dram: '15.5',
    dram_percent: '42.1',
    flops: '88.8',
    flops_percent: '73.2',
    math_fidelity: 'HiFi4',
    output_datatype: 'BFLOAT16',
    output_0_memory: '',
    input_0_datatype: 'BFLOAT16',
    input_1_datatype: 'BFLOAT16',
    dram_sharded: '',
    input_0_memory: 'DEV_0_DRAM_INTERLEAVED',
    input_1_memory: '',
    inner_dim_block_size: '',
    output_subblock_h: '',
    output_subblock_w: '',
    pm_ideal_ns: '1000',
    op_type: OpType.DEVICE_OP,
    hash: null,
    cache_hit: null,
} as unknown as PerfTableRow;

const REPORT_A = { path: '/reports/a', reportName: 'report-a' };
const REPORT_B = { path: '/reports/b', reportName: 'report-b' };
const SELECTED_ROW_ID = 100;
const MATH_FIDELITY = 'HiFi4';
const DURATION_BUCKET_MIN_US = 100;

afterEach(cleanup);

beforeEach(() => {
    perfReportProps.mockClear();
    chartsTabProps.mockClear();
    vi.mocked(enrichRowData).mockClear();
    // Loading state: the route keeps its shell mounted (no full-page spinner) and the
    // report shell is stubbed, so only the route's active-report effects run here.
    (usePerformanceReport as Mock).mockReturnValue({ data: undefined, isLoading: true, error: null });
    (usePerformanceComparisonReport as Mock).mockReturnValue({ data: undefined });
    (usePerfFolderList as Mock).mockReturnValue({ data: undefined });
    (usePerformanceRange as Mock).mockReturnValue(null);
    (useOpToPerfIdFiltered as Mock).mockReturnValue([]);
    (useL1PressureByOperation as Mock).mockReturnValue({ status: L1PressureStatus.Unavailable, data: null });
    (usePerfMeta as Mock).mockReturnValue({ data: undefined, isLoading: false });
    (usePerfMetas as Mock).mockReturnValue([]);
});

function formatFilterProbe(values: unknown[]): string {
    return values.length === 0 ? 'empty' : values.join(',');
}

function PerformanceController() {
    const [selected, setSelected] = useAtom(selectedPerfRowIdAtom);
    const [rawOpCodeFilterList, setRawOpCodeFilterList] = useAtom(rawOpCodeFilterListAtom);
    const [mathFilterList, setMathFilterList] = useAtom(mathFilterListAtom);
    const [bufferTypeFilterList, setBufferTypeFilterList] = useAtom(bufferTypeFilterListAtom);
    const [layoutFilterList, setLayoutFilterList] = useAtom(layoutFilterListAtom);
    const [durationBucketFilterList, setDurationBucketFilterList] = useAtom(durationBucketFilterListAtom);
    const setReport = useSetAtom(activePerformanceReportAtom);

    return (
        <div>
            <span data-testid='selected-row-probe'>{selected === null ? 'null' : String(selected)}</span>
            <span data-testid='raw-op-code-filter-probe'>{formatFilterProbe(rawOpCodeFilterList)}</span>
            <span data-testid='math-filter-probe'>{formatFilterProbe(mathFilterList)}</span>
            <span data-testid='buffer-type-filter-probe'>{formatFilterProbe(bufferTypeFilterList)}</span>
            <span data-testid='layout-filter-probe'>{formatFilterProbe(layoutFilterList)}</span>
            <span data-testid='duration-bucket-filter-probe'>{formatFilterProbe(durationBucketFilterList)}</span>
            <button
                type='button'
                data-testid='select-row'
                onClick={() => setSelected(SELECTED_ROW_ID)}
            >
                select
            </button>
            <button
                type='button'
                data-testid='set-all-filters'
                onClick={() => {
                    setRawOpCodeFilterList(['Matmul']);
                    setMathFilterList([MATH_FIDELITY]);
                    setBufferTypeFilterList([BufferType.L1]);
                    setLayoutFilterList([DeviceOperationLayoutTypes.INTERLEAVED]);
                    setDurationBucketFilterList([DURATION_BUCKET_MIN_US]);
                }}
            >
                filter
            </button>
            <button
                type='button'
                data-testid='set-report-a'
                onClick={() => setReport(REPORT_A)}
            >
                a
            </button>
            <button
                type='button'
                data-testid='set-report-b'
                onClick={() => setReport(REPORT_B)}
            >
                b
            </button>
        </div>
    );
}

describe('Performance route', () => {
    it('clears selectedPerfRowIdAtom when the active performance report changes', () => {
        render(
            <TestProviders>
                <Performance />
                <PerformanceController />
            </TestProviders>,
        );

        fireEvent.click(screen.getByTestId('set-report-a'));

        fireEvent.click(screen.getByTestId('select-row'));
        expect(screen.getByTestId('selected-row-probe')).toHaveTextContent(String(SELECTED_ROW_ID));

        fireEvent.click(screen.getByTestId('set-report-b'));

        expect(screen.getByTestId('selected-row-probe')).toHaveTextContent('null');
    });

    it('does not re-clear selectedPerfRowIdAtom while the active report is unchanged', () => {
        render(
            <TestProviders>
                <Performance />
                <PerformanceController />
            </TestProviders>,
        );

        fireEvent.click(screen.getByTestId('set-report-a'));
        fireEvent.click(screen.getByTestId('select-row'));
        expect(screen.getByTestId('selected-row-probe')).toHaveTextContent(String(SELECTED_ROW_ID));

        fireEvent.click(screen.getByTestId('set-report-a'));

        expect(screen.getByTestId('selected-row-probe')).toHaveTextContent(String(SELECTED_ROW_ID));
    });

    it('annotates enriched rows and passes maxCores to PerfReport', () => {
        (usePerformanceReport as Mock).mockReturnValue({
            data: { report: [DRAM_PERF_ROW], stacked_report: [], signposts: [] },
            isLoading: false,
            error: null,
        });
        (usePerformanceRange as Mock).mockReturnValue([1, 1]);

        render(
            <TestProviders
                initialAtomValues={[
                    [activePerformanceReportAtom, REPORT_A],
                    [selectedPerformanceRangeAtom, [1, 1]],
                ]}
            >
                <Performance />
            </TestProviders>,
        );

        expect(perfReportProps).toHaveBeenCalled();
        const lastProps = perfReportProps.mock.calls.at(-1)?.[0];
        expect(lastProps?.maxCores).toEqual(expect.any(Number));
        expect(lastProps?.data?.[0]?.heuristicFlags).toContain(PerfHeuristicFlag.DRAM_BOUND);
        expect(lastProps?.data?.[0]?.heuristicFlagDetails?.[PerfHeuristicFlag.DRAM_BOUND]).toBe('Bound: DRAM');
    });

    it('uses perfRange when selectedRange is not yet synced so the table does not flash empty', () => {
        (usePerformanceReport as Mock).mockReturnValue({
            data: { report: [DRAM_PERF_ROW], stacked_report: [], signposts: [] },
            isLoading: false,
            error: null,
        });
        (usePerformanceRange as Mock).mockReturnValue([1, 1]);

        render(
            <TestProviders
                initialAtomValues={[
                    [activePerformanceReportAtom, REPORT_A],
                    // selectedPerformanceRangeAtom defaults to null — the gap before RangeSlider syncs.
                ]}
            >
                <Performance />
            </TestProviders>,
        );

        const lastProps = perfReportProps.mock.calls.at(-1)?.[0];
        expect(lastProps?.isLoading).toBe(false);
        expect(lastProps?.data).toHaveLength(1);
        expect(lastProps?.data?.[0]?.heuristicFlags).toContain(PerfHeuristicFlag.DRAM_BOUND);
    });

    // The other half of `clampSelectionToRange`'s widen contract. Op<->perf ids now
    // resolve against the link-pinned report while this table spans the rows the tab
    // is showing, so a mapped selection can land entirely outside the report — here
    // it must fall back to the full span rather than render an empty table.
    //
    // The range has to have width for this to pin anything: over a zero-width range
    // the widen branch and the clamp branch both return that single id, so the
    // assertion could not tell them apart. With [1, 9] and two rows, widening keeps
    // both and clamping to the far edge would keep one.
    it('falls back to the whole report when the selected range misses it entirely', () => {
        (usePerformanceReport as Mock).mockReturnValue({
            data: {
                report: [DRAM_PERF_ROW, { ...DRAM_PERF_ROW, id: '9' }],
                stacked_report: [],
                signposts: [],
            },
            isLoading: false,
            error: null,
        });
        (usePerformanceRange as Mock).mockReturnValue([1, 9]);

        render(
            <TestProviders
                initialAtomValues={[
                    [activePerformanceReportAtom, REPORT_A],
                    // Left over from a signpost window that does not overlap this report.
                    [selectedPerformanceRangeAtom, [40, 50]],
                ]}
            >
                <Performance />
            </TestProviders>,
        );

        expect(perfReportProps.mock.calls.at(-1)?.[0]?.data).toHaveLength(2);
    });

    it('keeps the table loading while report rows exist but no range is available yet', () => {
        (usePerformanceReport as Mock).mockReturnValue({
            data: { report: [DRAM_PERF_ROW], stacked_report: [], signposts: [] },
            isLoading: false,
            error: null,
        });
        (usePerformanceRange as Mock).mockReturnValue(null);

        render(
            <TestProviders initialAtomValues={[[activePerformanceReportAtom, REPORT_A]]}>
                <Performance />
            </TestProviders>,
        );

        const lastProps = perfReportProps.mock.calls.at(-1)?.[0];
        expect(lastProps?.isLoading).toBe(true);
        expect(lastProps?.data).toHaveLength(0);
    });

    it('annotates comparison report rows with heuristic flags', () => {
        (usePerformanceReport as Mock).mockReturnValue({
            data: { report: [DRAM_PERF_ROW], stacked_report: [], signposts: [] },
            isLoading: false,
            error: null,
        });
        (usePerformanceComparisonReport as Mock).mockReturnValue({
            data: [{ report: [{ ...DRAM_PERF_ROW, id: '2' }], stacked_report: [] }],
        });
        (usePerformanceRange as Mock).mockReturnValue([1, 2]);
        (usePerfMetas as Mock).mockReturnValue([{ max_cores: 130, architecture: null, frequency: null }]);

        render(
            <TestProviders
                initialAtomValues={[
                    [activePerformanceReportAtom, REPORT_A],
                    [selectedPerformanceRangeAtom, [1, 2]],
                ]}
            >
                <Performance />
            </TestProviders>,
        );

        const lastProps = perfReportProps.mock.calls.at(-1)?.[0];
        expect(lastProps?.comparisonData?.[0]?.[0]?.heuristicFlags).toContain(PerfHeuristicFlag.DRAM_BOUND);
        expect(lastProps?.comparisonData?.[0]?.[0]?.heuristicFlagDetails?.[PerfHeuristicFlag.DRAM_BOUND]).toBe(
            'Bound: DRAM',
        );
        expect(lastProps?.comparisonMaxCores).toEqual([130]);
    });

    it('clears all table chip filters when the active performance report changes', () => {
        render(
            <TestProviders>
                <Performance />
                <PerformanceController />
            </TestProviders>,
        );

        fireEvent.click(screen.getByTestId('set-report-a'));
        fireEvent.click(screen.getByTestId('set-all-filters'));
        expect(screen.getByTestId('raw-op-code-filter-probe')).toHaveTextContent('Matmul');
        expect(screen.getByTestId('math-filter-probe')).toHaveTextContent(MATH_FIDELITY);
        expect(screen.getByTestId('buffer-type-filter-probe')).toHaveTextContent(String(BufferType.L1));
        expect(screen.getByTestId('layout-filter-probe')).toHaveTextContent(DeviceOperationLayoutTypes.INTERLEAVED);
        expect(screen.getByTestId('duration-bucket-filter-probe')).toHaveTextContent(String(DURATION_BUCKET_MIN_US));

        fireEvent.click(screen.getByTestId('set-report-b'));

        expect(screen.getByTestId('raw-op-code-filter-probe')).toHaveTextContent('empty');
        expect(screen.getByTestId('math-filter-probe')).toHaveTextContent('empty');
        expect(screen.getByTestId('buffer-type-filter-probe')).toHaveTextContent('empty');
        expect(screen.getByTestId('layout-filter-probe')).toHaveTextContent('empty');
        expect(screen.getByTestId('duration-bucket-filter-probe')).toHaveTextContent('empty');
    });

    it('does not re-clear table chip filters while the active report is unchanged', () => {
        render(
            <TestProviders>
                <Performance />
                <PerformanceController />
            </TestProviders>,
        );

        fireEvent.click(screen.getByTestId('set-report-a'));
        fireEvent.click(screen.getByTestId('set-all-filters'));
        expect(screen.getByTestId('raw-op-code-filter-probe')).toHaveTextContent('Matmul');
        expect(screen.getByTestId('math-filter-probe')).toHaveTextContent(MATH_FIDELITY);

        fireEvent.click(screen.getByTestId('set-report-a'));

        expect(screen.getByTestId('raw-op-code-filter-probe')).toHaveTextContent('Matmul');
        expect(screen.getByTestId('math-filter-probe')).toHaveTextContent(MATH_FIDELITY);
        expect(screen.getByTestId('buffer-type-filter-probe')).toHaveTextContent(String(BufferType.L1));
        expect(screen.getByTestId('layout-filter-probe')).toHaveTextContent(DeviceOperationLayoutTypes.INTERLEAVED);
        expect(screen.getByTestId('duration-bucket-filter-probe')).toHaveTextContent(String(DURATION_BUCKET_MIN_US));
    });

    it('does not clear table chip filters on Performance remount with the same report', () => {
        function RemountHarness() {
            const [routeKey, setRouteKey] = useState(0);

            return (
                <>
                    <Performance key={routeKey} />
                    <PerformanceController />
                    <button
                        type='button'
                        data-testid='remount-performance'
                        onClick={() => setRouteKey((key) => key + 1)}
                    >
                        remount
                    </button>
                </>
            );
        }

        render(
            <TestProviders>
                <RemountHarness />
            </TestProviders>,
        );

        fireEvent.click(screen.getByTestId('set-report-a'));
        fireEvent.click(screen.getByTestId('set-all-filters'));
        expect(screen.getByTestId('raw-op-code-filter-probe')).toHaveTextContent('Matmul');

        fireEvent.click(screen.getByTestId('remount-performance'));

        expect(screen.getByTestId('raw-op-code-filter-probe')).toHaveTextContent('Matmul');
        expect(screen.getByTestId('math-filter-probe')).toHaveTextContent(MATH_FIDELITY);
        expect(screen.getByTestId('buffer-type-filter-probe')).toHaveTextContent(String(BufferType.L1));
        expect(screen.getByTestId('layout-filter-probe')).toHaveTextContent(DeviceOperationLayoutTypes.INTERLEAVED);
        expect(screen.getByTestId('duration-bucket-filter-probe')).toHaveTextContent(String(DURATION_BUCKET_MIN_US));
    });

    it('does not clear selectedPerfRowIdAtom on Performance remount with the same report', () => {
        function RemountHarness() {
            const [routeKey, setRouteKey] = useState(0);

            return (
                <>
                    <Performance key={routeKey} />
                    <PerformanceController />
                    <button
                        type='button'
                        data-testid='remount-performance'
                        onClick={() => setRouteKey((key) => key + 1)}
                    >
                        remount
                    </button>
                </>
            );
        }

        render(
            <TestProviders>
                <RemountHarness />
            </TestProviders>,
        );

        fireEvent.click(screen.getByTestId('set-report-a'));
        fireEvent.click(screen.getByTestId('select-row'));
        expect(screen.getByTestId('selected-row-probe')).toHaveTextContent(String(SELECTED_ROW_ID));

        fireEvent.click(screen.getByTestId('remount-performance'));

        expect(screen.getByTestId('selected-row-probe')).toHaveTextContent(String(SELECTED_ROW_ID));
    });
});

const reportWith = (opCodes: string[]) => ({
    report: opCodes.map((opCode, index) => ({
        ...DRAM_PERF_ROW,
        id: String(index + 1),
        op_code: opCode,
        raw_op_code: opCode,
    })),
    stacked_report: [],
    signposts: [],
});

const mockReport = (opCodes: string[]) => {
    const data = reportWith(opCodes);

    (usePerformanceReport as Mock).mockReturnValue({ data, isLoading: false, error: null });
    (usePerformanceRange as Mock).mockReturnValue([1, opCodes.length]);

    return data;
};

interface ChartsTabProps {
    selectedOpCodes: Marker[];
    opCodeOptions: Marker[];
    filteredPerfData: TypedPerfTableRow[];
    updateOpCodes: (opCodes: Marker[]) => void;
}

const chartsTabCalls = () => chartsTabProps.mock.calls.map(([props]) => props as ChartsTabProps);
const lastChartsTab = () => chartsTabCalls().at(-1) as ChartsTabProps;
const opCodesOf = (markers: { opCode?: string; raw_op_code?: string }[]) =>
    markers.map((marker) => marker.opCode ?? marker.raw_op_code);

const renderCharts = () =>
    render(
        <TestProviders
            initialAtomValues={[
                [activePerformanceReportAtom, REPORT_A],
                [perfSelectedTabAtom, PerfTabIds.CHARTS],
            ]}
        >
            <Performance />
        </TestProviders>,
    );

const selectOpCodes = (opCodes: string[]) =>
    act(() => {
        const { opCodeOptions, updateOpCodes } = lastChartsTab();

        updateOpCodes(opCodeOptions.filter((option) => opCodes.includes(option.opCode)));
    });

describe('Performance route chart op-code selection', () => {
    // It used to start empty and be filled in by an effect after the first commit, so every
    // report paid for a second render of the whole route before the charts settled.
    it('selects every op code from the first render', () => {
        mockReport(['Matmul', 'Conv2d']);

        renderCharts();

        expect(chartsTabCalls().length).toBeGreaterThan(0);

        for (const props of chartsTabCalls()) {
            expect(opCodesOf(props.selectedOpCodes)).toEqual(['Matmul', 'Conv2d']);
            expect(opCodesOf(props.filteredPerfData)).toEqual(['Matmul', 'Conv2d']);
        }
    });

    it('keeps the user’s selection while the options stay the same', () => {
        mockReport(['Matmul', 'Conv2d']);
        const { rerender } = renderCharts();

        selectOpCodes(['Matmul']);
        expect(opCodesOf(lastChartsTab().selectedOpCodes)).toEqual(['Matmul']);
        expect(opCodesOf(lastChartsTab().filteredPerfData)).toEqual(['Matmul']);

        // A new response carrying the same op codes, as a view toggle refetches
        mockReport(['Matmul', 'Conv2d']);
        rerender(
            <TestProviders>
                <Performance />
            </TestProviders>,
        );

        expect(opCodesOf(lastChartsTab().selectedOpCodes)).toEqual(['Matmul']);
        expect(opCodesOf(lastChartsTab().filteredPerfData)).toEqual(['Matmul']);
    });

    it('starts from every op code again when the options change', () => {
        mockReport(['Matmul', 'Conv2d']);
        const { rerender } = renderCharts();

        selectOpCodes(['Matmul']);
        mockReport(['Matmul', 'Conv2d', 'Softmax']);
        rerender(
            <TestProviders>
                <Performance />
            </TestProviders>,
        );

        expect(opCodesOf(lastChartsTab().selectedOpCodes)).toEqual(['Matmul', 'Conv2d', 'Softmax']);
        expect(opCodesOf(lastChartsTab().filteredPerfData)).toEqual(['Matmul', 'Conv2d', 'Softmax']);
    });

    it('does not revive a selection when the options it was made from come back', () => {
        mockReport(['Matmul', 'Conv2d']);
        const { rerender } = renderCharts();

        selectOpCodes(['Matmul']);
        mockReport(['Softmax']);
        rerender(
            <TestProviders>
                <Performance />
            </TestProviders>,
        );
        mockReport(['Matmul', 'Conv2d']);
        rerender(
            <TestProviders>
                <Performance />
            </TestProviders>,
        );

        expect(opCodesOf(lastChartsTab().selectedOpCodes)).toEqual(['Matmul', 'Conv2d']);
    });

    it('charts no rows once the user clears every op code', () => {
        mockReport(['Matmul', 'Conv2d']);
        renderCharts();

        selectOpCodes([]);

        expect(lastChartsTab().selectedOpCodes).toEqual([]);
        expect(lastChartsTab().filteredPerfData).toEqual([]);
    });
});

function RangeController({ range }: { range: [number, number] }) {
    const setRange = useSetAtom(selectedPerformanceRangeAtom);

    return (
        <button
            type='button'
            data-testid='set-range'
            // A new tuple on every click, as the slider's clamp produces
            onClick={() => setRange([range[0], range[1]])}
        >
            range
        </button>
    );
}

describe('Performance route table range', () => {
    it('enriches the report’s own rows when the range spans the whole report', () => {
        const { report } = mockReport(['Matmul', 'Conv2d', 'Softmax']);

        render(
            <TestProviders
                initialAtomValues={[
                    [activePerformanceReportAtom, REPORT_A],
                    [selectedPerformanceRangeAtom, [1, 3]],
                ]}
            >
                <Performance />
            </TestProviders>,
        );

        expect(vi.mocked(enrichRowData).mock.calls.at(-1)?.[0]).toBe(report);
    });

    it('filters to the selected rows when the range covers part of the report', () => {
        mockReport(['Matmul', 'Conv2d', 'Softmax']);

        render(
            <TestProviders
                initialAtomValues={[
                    [activePerformanceReportAtom, REPORT_A],
                    [selectedPerformanceRangeAtom, [2, 3]],
                ]}
            >
                <Performance />
            </TestProviders>,
        );

        expect(opCodesOf(perfReportProps.mock.calls.at(-1)?.[0]?.data)).toEqual(['Conv2d', 'Softmax']);
    });

    // The slider re-sets its atom to the bounds it already holds as it syncs, and each is a
    // new tuple. Followed by identity, every one re-enriched the report and re-rendered the table.
    it.each([
        ['the whole report', [1, 3] as [number, number]],
        ['part of the report', [2, 3] as [number, number]],
    ])('keeps the rows it hands the table when %s is selected again', (_label, range) => {
        mockReport(['Matmul', 'Conv2d', 'Softmax']);

        render(
            <TestProviders
                initialAtomValues={[
                    [activePerformanceReportAtom, REPORT_A],
                    [selectedPerformanceRangeAtom, range],
                ]}
            >
                <Performance />
                <RangeController range={range} />
            </TestProviders>,
        );

        const rowsBefore = perfReportProps.mock.calls.at(-1)?.[0]?.data;
        const enrichmentsBefore = vi.mocked(enrichRowData).mock.calls.length;

        fireEvent.click(screen.getByTestId('set-range'));

        expect(perfReportProps.mock.calls.at(-1)?.[0]?.data).toBe(rowsBefore);
        expect(vi.mocked(enrichRowData).mock.calls.length).toBe(enrichmentsBefore);
    });
});
