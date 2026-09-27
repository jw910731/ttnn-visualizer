// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import '@testing-library/jest-dom/vitest';
import type { Virtualizer } from '@tanstack/react-virtual';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { Mock, afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import PerfTable from '../src/components/performance/PerfTable';
import { PERF_HEURISTIC_FLAG_DEFINITIONS, PerfHeuristicFlag } from '../src/definitions/PerfHeuristics';
import { OpType } from '../src/definitions/Performance';
import { TEST_IDS } from '../src/definitions/TestIds';
import { getRowGroupKeys } from '../src/functions/perfFunctions';
import { DEFAULT_MAX_CORES } from '../src/functions/getCoreCount';
import { useGetNPEManifest, useOpToPerfIdFiltered, useOperationsList, usePerfMeta } from '../src/hooks/useAPI';
import { TypedPerfTableRow, signpostRowDefaults } from '../src/model/PerfTable';
import { TestProviders } from './helpers/TestProviders';

const formatCellSpy = vi.hoisted(() => vi.fn());
const virtualizerProbe = vi.hoisted(() => ({ current: null as Virtualizer<Window, Element> | null }));

vi.mock('../src/hooks/useAPI.tsx', () => ({
    useGetNPEManifest: vi.fn(),
    useOpToPerfIdFiltered: vi.fn(),
    useOperationsList: vi.fn(),
    usePerfMeta: vi.fn(),
}));

// Counts cell renders, which is what a row group re-rendering costs
vi.mock('../src/functions/perfFunctions', async (importOriginal) => {
    const actual = await importOriginal<typeof import('../src/functions/perfFunctions')>();

    return {
        ...actual,
        formatCell: (...args: Parameters<typeof actual.formatCell>) => {
            formatCellSpy(args[0]);
            return actual.formatCell(...args);
        },
    };
});

// Exposes the table's virtualiser so a spec can check the spacers against its own model of the list
vi.mock('@tanstack/react-virtual', async (importOriginal) => {
    const actual = await importOriginal<typeof import('@tanstack/react-virtual')>();

    return {
        ...actual,
        useWindowVirtualizer: (...args: Parameters<typeof actual.useWindowVirtualizer>) => {
            const virtualizer = actual.useWindowVirtualizer(...args);

            virtualizerProbe.current = virtualizer;

            return virtualizer;
        },
    };
});

// jsdom lays nothing out, so a measured group would be 0px tall and the virtualiser would keep
// widening its window until every row was rendered.
const ROW_GROUP_HEIGHT_PX = 30;
const LARGE_REPORT_ROW_COUNT = 5000;
// 768px of jsdom viewport at 30px a group, plus overscan either side, with headroom
const MAX_RENDERED_GROUPS = 80;
const SCROLLED_TO_INDEX = 3000;
// Where the list starts in the document, below the page header, toolbar and table header
const LIST_OFFSET_PX = 600;

const EMPTY_COMPARISON_DATA: TypedPerfTableRow[][] = [];
const NPE_MANIFEST = { data: [], error: null };
const OPERATIONS = { data: [] };
const PERF_META = { data: null, isLoading: false };

const deviceRow = (id: number, overrides: Partial<TypedPerfTableRow> = {}): TypedPerfTableRow =>
    ({
        id,
        op: id,
        op_type: OpType.DEVICE_OP,
        op_code: `Op${id}`,
        raw_op_code: `Op${id}`,
        advice: [],
        bound: null,
        isFirstHashOccurrence: true,
        ...overrides,
    }) as unknown as TypedPerfTableRow;

const signpostRow = (id: number): TypedPerfTableRow =>
    ({
        ...signpostRowDefaults,
        id,
        op_code: '---SIGNPOST---',
        raw_op_code: '---SIGNPOST---',
    }) as unknown as TypedPerfTableRow;

const makeRows = (count: number) => Array.from({ length: count }, (_, index) => deviceRow(index));

interface TableProps {
    data: TypedPerfTableRow[];
    comparisonData?: TypedPerfTableRow[][];
    activeReportComparisonIndex?: number | null;
    provideMatmulAdvice?: boolean;
}

const renderTableElement = ({
    data,
    comparisonData = EMPTY_COMPARISON_DATA,
    activeReportComparisonIndex = null,
    provideMatmulAdvice = false,
}: TableProps) => (
    <TestProviders>
        <PerfTable
            data={data}
            comparisonData={comparisonData}
            filters={null}
            provideMatmulAdvice={provideMatmulAdvice}
            hiliteHighDispatch={false}
            reportFolderName='unit-test'
            maxCores={DEFAULT_MAX_CORES}
            activeReportComparisonIndex={activeReportComparisonIndex}
        />
    </TestProviders>
);

const getRenderedGroups = () =>
    Array.from(screen.getByRole('table').querySelectorAll<HTMLElement>('tbody.perf-table-row-group'));

const getRenderedIndices = () => getRenderedGroups().map((group) => Number(group.dataset.index));

const getRenderedRowIds = () => new Set(formatCellSpy.mock.calls.map(([row]) => (row as TypedPerfTableRow).id));

const getSpacerHeights = () =>
    Array.from(screen.getByRole('table').querySelectorAll<HTMLElement>('tbody.perf-table-spacer tr'), (spacer) =>
        parseFloat(spacer.style.height),
    );

const getVirtualizer = () => {
    if (!virtualizerProbe.current) {
        throw new Error('The table has not created a virtualiser');
    }

    return virtualizerProbe.current;
};

const setScrollY = (scrollY: number) =>
    Object.defineProperty(window, 'scrollY', { configurable: true, writable: true, value: scrollY });

beforeEach(() => {
    Object.defineProperty(HTMLTableSectionElement.prototype, 'offsetHeight', {
        configurable: true,
        get: () => ROW_GROUP_HEIGHT_PX,
    });
    // The virtualiser re-asserts the scroll position on mount; jsdom does not implement scrolling
    vi.spyOn(window, 'scrollTo').mockImplementation(() => {});

    (useGetNPEManifest as Mock).mockReturnValue(NPE_MANIFEST);
    (useOperationsList as Mock).mockReturnValue(OPERATIONS);
    (usePerfMeta as Mock).mockReturnValue(PERF_META);
    (useOpToPerfIdFiltered as Mock).mockReturnValue([{ opId: 1, perfId: '1' }]);
    formatCellSpy.mockClear();
    virtualizerProbe.current = null;
});

afterEach(() => {
    cleanup();
    Reflect.deleteProperty(HTMLTableSectionElement.prototype, 'offsetHeight');
    setScrollY(0);
    vi.restoreAllMocks();
});

describe('PerfTable virtualisation', () => {
    it('renders a bounded window of a large report rather than every row', () => {
        render(renderTableElement({ data: makeRows(LARGE_REPORT_ROW_COUNT) }));

        const table = screen.getByRole('table');
        const groups = getRenderedGroups();

        expect(groups.length).toBeGreaterThan(0);
        expect(groups.length).toBeLessThan(MAX_RENDERED_GROUPS);
        // One row per group, plus the sizer, spacers, header and footer
        expect(table.querySelectorAll('tr').length).toBeLessThan(MAX_RENDERED_GROUPS + 5);
        expect(getRenderedIndices()[0]).toBe(0);
        expect(screen.getByText('Op0')).toBeInTheDocument();
        expect(screen.queryByText(`Op${LARGE_REPORT_ROW_COUNT - 1}`)).not.toBeInTheDocument();
    });

    it('stands spacer rows in for the groups it does not render', () => {
        render(renderTableElement({ data: makeRows(LARGE_REPORT_ROW_COUNT) }));

        const spacers = screen.getByRole('table').querySelectorAll<HTMLElement>('tbody.perf-table-spacer tr');
        const bottomSpacer = spacers[spacers.length - 1];

        expect(spacers).toHaveLength(2);
        expect(parseFloat(spacers[0].style.height)).toBe(0);
        expect(parseFloat(bottomSpacer.style.height)).toBeGreaterThan(
            (LARGE_REPORT_ROW_COUNT - MAX_RENDERED_GROUPS) * ROW_GROUP_HEIGHT_PX,
        );
    });

    it('moves the window with the page scroll', () => {
        render(renderTableElement({ data: makeRows(LARGE_REPORT_ROW_COUNT) }));

        act(() => {
            setScrollY(SCROLLED_TO_INDEX * ROW_GROUP_HEIGHT_PX);
            window.dispatchEvent(new Event('scroll'));
        });

        const indices = getRenderedIndices();
        const topSpacer = screen.getByRole('table').querySelector<HTMLElement>('tbody.perf-table-spacer tr')!;

        expect(indices.length).toBeLessThan(MAX_RENDERED_GROUPS);
        // Unmeasured groups are estimated a little taller than the stubbed height, so the window
        // lands near, not exactly on, the scrolled-to row
        expect(Math.min(...indices)).toBeGreaterThan(SCROLLED_TO_INDEX / 2);
        expect(Math.max(...indices)).toBeLessThan(LARGE_REPORT_ROW_COUNT);
        expect(parseFloat(topSpacer.style.height)).toBeGreaterThan(0);
        expect(screen.queryByText('Op0')).not.toBeInTheDocument();
    });

    it('sizes columns from rows outside the rendered window', () => {
        const longOpCode = 'AnUnusuallyLongOperationCodeFarBelowTheFold';
        const rows = makeRows(LARGE_REPORT_ROW_COUNT);
        rows[LARGE_REPORT_ROW_COUNT - 10] = deviceRow(LARGE_REPORT_ROW_COUNT - 10, {
            op_code: longOpCode,
            raw_op_code: longOpCode,
        });

        render(renderTableElement({ data: rows }));

        const sizer = screen.getByRole('table').querySelector('tbody.perf-table-sizer')!;

        expect(screen.queryByText(longOpCode)).not.toBeInTheDocument();
        expect(sizer.querySelector(`[data-text="${longOpCode}"]`)).not.toBeNull();
        // Generated content only, so the sizer never answers a text query or reaches assistive tech
        expect(sizer).toHaveAttribute('aria-hidden', 'true');
        expect(sizer.textContent).toBe('');
    });

    it('sizes the flags column from the flags each row carries, never flags from different rows together', () => {
        const rows = makeRows(LARGE_REPORT_ROW_COUNT);
        rows[0] = deviceRow(0, { heuristicFlags: [PerfHeuristicFlag.DRAM_BOUND] });
        rows[LARGE_REPORT_ROW_COUNT - 1] = deviceRow(LARGE_REPORT_ROW_COUNT - 1, {
            heuristicFlags: [PerfHeuristicFlag.LOW_UTILISATION],
        });

        render(renderTableElement({ data: rows }));

        const sizer = screen.getByRole('table').querySelector('tbody.perf-table-sizer')!;
        const flagSamples = Array.from(sizer.querySelectorAll('.perf-heuristic-flags'), (sample) =>
            Array.from(sample.querySelectorAll('[data-text]'), (label) => label.getAttribute('data-text')),
        );

        expect(flagSamples).toEqual([
            [PERF_HEURISTIC_FLAG_DEFINITIONS[PerfHeuristicFlag.DRAM_BOUND].shortLabel],
            [PERF_HEURISTIC_FLAG_DEFINITIONS[PerfHeuristicFlag.LOW_UTILISATION].shortLabel],
        ]);
    });
});

describe('PerfTable scroll margin', () => {
    // The list-start spacer is the element whose document offset the table tracks
    const stubListOffset = () =>
        vi
            .spyOn(HTMLTableSectionElement.prototype, 'getBoundingClientRect')
            .mockImplementation(() => ({ top: LIST_OFFSET_PX - window.scrollY }) as DOMRect);

    it('starts the list at its offset in the document rather than at the top of the page', () => {
        render(renderTableElement({ data: makeRows(LARGE_REPORT_ROW_COUNT) }));
        const groupCountWithoutOffset = getRenderedGroups().length;
        cleanup();

        stubListOffset();
        render(renderTableElement({ data: makeRows(LARGE_REPORT_ROW_COUNT) }));

        expect(getVirtualizer().options.scrollMargin).toBe(LIST_OFFSET_PX);
        expect(getRenderedIndices()[0]).toBe(0);
        expect(getSpacerHeights()[0]).toBe(0);
        // The offset pushes the list partly below the fold, so fewer groups are in view
        expect(getRenderedGroups().length).toBeLessThan(groupCountWithoutOffset);
    });

    it('sizes the spacers from the start of the list, not the top of the document', () => {
        stubListOffset();
        render(renderTableElement({ data: makeRows(LARGE_REPORT_ROW_COUNT) }));

        act(() => {
            setScrollY(LIST_OFFSET_PX + SCROLLED_TO_INDEX * ROW_GROUP_HEIGHT_PX);
            window.dispatchEvent(new Event('scroll'));
        });

        const virtualizer = getVirtualizer();
        const virtualItems = virtualizer.getVirtualItems();
        const [topSpacerHeight, bottomSpacerHeight] = getSpacerHeights();
        const renderedHeight = virtualItems.reduce((total, item) => total + item.size, 0);

        expect(getRenderedIndices()).toEqual(virtualItems.map((item) => item.index));
        // Item positions include the offset; the spacers sit inside the list, so they must not
        expect(topSpacerHeight).toBe(virtualItems[0].start - LIST_OFFSET_PX);
        expect(topSpacerHeight + renderedHeight + bottomSpacerHeight).toBe(virtualizer.getTotalSize());
    });
});

describe('PerfTable virtualiser identity', () => {
    it('keys items by row alone, so a key lookup never grows with the column count', () => {
        const rows = [signpostRow(4), ...makeRows(10), signpostRow(4)];

        render(renderTableElement({ data: rows }));

        const { getItemKey } = getVirtualizer().options;

        expect(rows.map((_, index) => getItemKey(index))).toEqual(getRowGroupKeys(rows));
    });

    it('keeps one virtualiser across re-renders and starts a new one for a new layout', () => {
        const rows = makeRows(20);
        const { rerender } = render(renderTableElement({ data: rows }));
        const firstVirtualizer = getVirtualizer();

        rerender(renderTableElement({ data: rows }));

        expect(getVirtualizer()).toBe(firstVirtualizer);

        // Heights measured without advice rows or comparison sub-rows are wrong for groups that have them
        rerender(renderTableElement({ data: rows, provideMatmulAdvice: true }));
        const adviceVirtualizer = getVirtualizer();

        expect(adviceVirtualizer).not.toBe(firstVirtualizer);

        rerender(renderTableElement({ data: rows, provideMatmulAdvice: true, comparisonData: [rows] }));

        expect(getVirtualizer()).not.toBe(adviceVirtualizer);
    });
});

describe('PerfTable row group stability', () => {
    it('re-renders only the groups whose selection changed', () => {
        const rows = makeRows(20);

        render(renderTableElement({ data: rows }));

        const triggers = screen.getAllByTestId(TEST_IDS.PERF_TENSOR_DRAWER_OPEN_BUTTON);

        formatCellSpy.mockClear();
        fireEvent.click(triggers[3]);

        expect(triggers[3].closest('tr')).toHaveClass('is-selected');
        expect(getRenderedRowIds()).toEqual(new Set([rows[3].id]));

        formatCellSpy.mockClear();
        fireEvent.click(triggers[5]);

        expect(triggers[3].closest('tr')).not.toHaveClass('is-selected');
        expect(triggers[5].closest('tr')).toHaveClass('is-selected');
        expect(getRenderedRowIds()).toEqual(new Set([rows[3].id, rows[5].id]));
    });

    it('does not re-render any group when the table re-renders with the same props', () => {
        const props = { data: makeRows(20) };
        const { rerender } = render(renderTableElement(props));

        formatCellSpy.mockClear();
        rerender(renderTableElement(props));

        expect(formatCellSpy).not.toHaveBeenCalled();
    });

    it('selects the comparison sub-row that holds the active report, not the primary row', () => {
        const comparisonReportRow = deviceRow(99, { op: undefined });
        const activeReportRow = deviceRow(1);

        render(
            renderTableElement({
                data: [comparisonReportRow],
                comparisonData: [[activeReportRow]],
                activeReportComparisonIndex: 0,
            }),
        );

        const trigger = screen.getByTestId(TEST_IDS.PERF_TENSOR_DRAWER_OPEN_BUTTON);
        fireEvent.click(trigger);

        const group = getRenderedGroups()[0];
        const [primaryRow, subRow] = within(group).getAllByRole('row');

        expect(subRow).toHaveClass('comparison-row', 'is-selected');
        expect(primaryRow).not.toHaveClass('is-selected');
    });
});

describe('PerfTable row keys', () => {
    it('keeps a signpost placeholder and the signpost it stands for as separate groups', () => {
        const consoleError = vi.spyOn(console, 'error');

        render(renderTableElement({ data: [signpostRow(4), deviceRow(1), signpostRow(4)] }));

        expect(screen.getAllByText('---SIGNPOST---')).toHaveLength(2);
        expect(consoleError.mock.calls.some(([message]) => String(message).includes('same key'))).toBe(false);
    });

    it('suffixes repeated identities and leaves the rest untouched', () => {
        const keys = getRowGroupKeys([signpostRow(4), deviceRow(4), signpostRow(4), deviceRow(5), signpostRow(4)]);

        expect(new Set(keys).size).toBe(keys.length);
        expect(keys[1]).toBe(`${OpType.DEVICE_OP}:4`);
        expect(keys[0]).not.toBe(keys[2]);
    });
});
