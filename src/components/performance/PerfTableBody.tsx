// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { useWindowVirtualizer } from '@tanstack/react-virtual';
import { useCallback, useMemo } from 'react';
import { ColumnDefinition, PerfTableFilters } from '../../definitions/PerfTable';
import { OperationsById, getRowGroupKeys, hasMatmulAdvice } from '../../functions/perfFunctions';
import useDocumentOffsetTop from '../../hooks/useDocumentOffsetTop';
import { PerfTableSizing } from '../../hooks/usePerfTableColumnSizing';
import { TypedPerfTableRow } from '../../model/PerfTable';
import PerfTableRowGroup from './PerfTableRowGroup';
import PerfTableSizer from './PerfTableSizer';

// Height of a row holding the small tensor-trigger button. Only a starting point: every rendered group is measured.
const ESTIMATED_ROW_HEIGHT_PX = 33;
const ESTIMATED_ADVICE_LINE_HEIGHT_PX = 18;
// Groups rendered beyond each edge of the viewport, so a fast scroll does not outrun rendering.
const ROW_GROUP_OVERSCAN = 15;

interface PerfTableBodyProps {
    rows: TypedPerfTableRow[];
    comparisonDatasets: TypedPerfTableRow[][];
    sizing: PerfTableSizing;
    visibleColumns: ColumnDefinition[];
    filters: PerfTableFilters;
    operationsById: OperationsById | null;
    npeFileByGlobalCallCount: ReadonlyMap<number, string>;
    firstRowOfOpRun: ReadonlySet<TypedPerfTableRow>;
    selectedPerfRowId: number | null;
    isPrimaryActiveReport: boolean;
    activeReportComparisonIndex: number | null;
    isReportsSynced: boolean;
    provideMatmulAdvice: boolean;
    onSelectRow: (rowId: number | null) => void;
    onOpenNpeTimeline: (file: string) => void;
}

/**
 * Renders only the row groups near the viewport, between spacer rows standing in for the rest. The
 * document is the scroll container on the performance page, so the list is offset by the table's
 * position in it rather than scrolled inside an element of its own.
 *
 * Measured heights hold for one layout only, since a group gains or loses rows and columns with it, so
 * the parent keys this component by layout and each layout starts with a virtualiser of its own.
 */
function PerfTableBody({
    rows,
    comparisonDatasets,
    sizing,
    visibleColumns,
    filters,
    operationsById,
    npeFileByGlobalCallCount,
    firstRowOfOpRun,
    selectedPerfRowId,
    isPrimaryActiveReport,
    activeReportComparisonIndex,
    isReportsSynced,
    provideMatmulAdvice,
    onSelectRow,
    onOpenNpeTimeline,
}: PerfTableBodyProps) {
    const [setListStartElement, scrollMargin] = useDocumentOffsetTop<HTMLTableSectionElement>();
    const rowKeys = useMemo(() => getRowGroupKeys(rows), [rows]);
    // The virtualiser looks up a key for every row after each re-measure, so it reads strings built once
    // per row set, whose hashes the engine caches, rather than composing a new one per call.
    const getItemKey = useCallback((index: number) => rowKeys[index], [rowKeys]);

    const estimateSize = useCallback(
        (index: number) => {
            const adviceLineCount = provideMatmulAdvice && hasMatmulAdvice(rows[index]) ? rows[index].advice.length : 0;

            return (
                ESTIMATED_ROW_HEIGHT_PX * (1 + comparisonDatasets.length) +
                ESTIMATED_ADVICE_LINE_HEIGHT_PX * adviceLineCount
            );
        },
        [rows, comparisonDatasets.length, provideMatmulAdvice],
    );

    const virtualizer = useWindowVirtualizer<HTMLTableSectionElement>({
        count: rows.length,
        estimateSize,
        getItemKey,
        overscan: ROW_GROUP_OVERSCAN,
        scrollMargin,
    });

    const virtualItems = virtualizer.getVirtualItems();
    const firstItem = virtualItems[0];
    const lastItem = virtualItems[virtualItems.length - 1];
    const paddingTop = firstItem ? firstItem.start - scrollMargin : 0;
    const paddingBottom = lastItem ? virtualizer.getTotalSize() - (lastItem.end - scrollMargin) : 0;
    const columnCount = visibleColumns.length + 1;
    const activeReportRows = isPrimaryActiveReport ? rows : comparisonDatasets[activeReportComparisonIndex ?? -1];

    return (
        <>
            <PerfTableSizer sizing={sizing} />

            {/* Always rendered: its position is where the list starts, which the scroll margin tracks */}
            <tbody
                ref={setListStartElement}
                className='perf-table-spacer'
                aria-hidden='true'
            >
                <tr style={{ height: paddingTop }}>
                    <td colSpan={columnCount} />
                </tr>
            </tbody>

            {virtualItems.map((virtualItem) => {
                const row = rows[virtualItem.index];
                const activeReportRow = activeReportRows?.[virtualItem.index];

                return (
                    <PerfTableRowGroup
                        key={virtualItem.key}
                        row={row}
                        index={virtualItem.index}
                        comparisonDatasets={comparisonDatasets}
                        visibleColumns={visibleColumns}
                        filters={filters}
                        operationsById={operationsById}
                        npeFileByGlobalCallCount={npeFileByGlobalCallCount}
                        isFirstOfOpRun={row.op === undefined || firstRowOfOpRun.has(row)}
                        isSelected={activeReportRow !== undefined && activeReportRow.id === selectedPerfRowId}
                        isPrimaryActiveReport={isPrimaryActiveReport}
                        activeReportComparisonIndex={activeReportComparisonIndex}
                        isReportsSynced={isReportsSynced}
                        provideMatmulAdvice={provideMatmulAdvice}
                        onSelectRow={onSelectRow}
                        onOpenNpeTimeline={onOpenNpeTimeline}
                        measureElement={virtualizer.measureElement}
                    />
                );
            })}

            {paddingBottom > 0 && (
                <tbody
                    className='perf-table-spacer'
                    aria-hidden='true'
                >
                    <tr style={{ height: paddingBottom }}>
                        <td colSpan={columnCount} />
                    </tr>
                </tbody>
            )}
        </>
    );
}

export default PerfTableBody;
