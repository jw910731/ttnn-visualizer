// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import { Button, ButtonVariant, Icon, Intent, Size } from '@blueprintjs/core';
import { IconNames } from '@blueprintjs/icons';
import classNames from 'classnames';
import { useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router';
import 'styles/components/PerfReport.scss';
import { ColumnKeys, PerfTableFilters } from '../../definitions/PerfTable';
import { TypedPerfTableRow } from '../../model/PerfTable';
import ROUTES from '../../definitions/Routes';
import { formatPercentage, formatSize } from '../../functions/math';
import { getOperationsById, isHostOp } from '../../functions/perfFunctions';
import { getEligiblePerfColumns, getFooterColumns, getVisiblePerfColumns } from '../../functions/perfTableColumns';
import { useGetNPEManifest, useOpToPerfIdFiltered, useOperationsSummary } from '../../hooks/useAPI';
import usePerfTableColumnSizing from '../../hooks/usePerfTableColumnSizing';
import useSortTable, { SortingDirection } from '../../hooks/useSortTable';
import { NPEManifestEntry } from '../../model/NPEModel';
import { hiddenPerfTableColumnsAtom, hideHostOpsAtom, mergeDevicesAtom, selectedPerfRowIdAtom } from '../../store/app';
import PerfDeviceArchitecture from './PerfDeviceArchitecture';
import PerfMultiDeviceNotice from './PerfMultiDeviceNotice';
import PerfTableBody from './PerfTableBody';
import PerfTableSkeleton from './PerfTableSkeleton';
import PerfTensorDrawer from './PerfTensorDrawer';
import PerfTableToolbar from './PerfTableToolbar';

const EMPTY_ROWS: TypedPerfTableRow[] = [];
const EMPTY_COMPARISON_DATA: TypedPerfTableRow[][] = [];
const EMPTY_NPE_MANIFEST: NPEManifestEntry[] = [];

// A call count listed more than once links to its first manifest entry.
const getNpeFileByGlobalCallCount = (npeManifest: NPEManifestEntry[] | null | undefined) => {
    const npeFileByGlobalCallCount = new Map<number, string>();

    for (const entry of npeManifest ?? EMPTY_NPE_MANIFEST) {
        if (!npeFileByGlobalCallCount.has(entry.global_call_count)) {
            npeFileByGlobalCallCount.set(entry.global_call_count, entry.file);
        }
    }

    return npeFileByGlobalCallCount;
};

interface PerformanceTableProps {
    data: TypedPerfTableRow[];
    comparisonData?: TypedPerfTableRow[][];
    filters: PerfTableFilters;
    provideMatmulAdvice: boolean;
    hiliteHighDispatch: boolean;
    /** On-disk folder of the report these rows came from; addresses the device-meta fetch. */
    reportFolderName: string | null;
    maxCores: number;
    hasL1PressureData?: boolean;
    isLoading?: boolean;
    // Identifies which comparison dataset holds the active profiler report's rows. The
    // tensor-drawer trigger only renders on those rows, since op-id sync from
    // `useOpToPerfIdFiltered()` and the drawer's tensor lookups in `useOperationsList()`
    // are both keyed to the active report. When `null` (default), `data` is the active report
    // and triggers render on the primary rows.
    activeReportComparisonIndex?: number | null;
}

const PerformanceTable = ({
    data,
    comparisonData,
    filters,
    provideMatmulAdvice,
    hiliteHighDispatch,
    reportFolderName,
    maxCores,
    hasL1PressureData = false,
    isLoading = false,
    activeReportComparisonIndex = null,
}: PerformanceTableProps) => {
    const hideHostOps = useAtomValue(hideHostOpsAtom);
    const hiddenColumns = useAtomValue(hiddenPerfTableColumnsAtom);
    const mergeDevices = useAtomValue(mergeDevicesAtom);
    const selectedPerfRowId = useAtomValue(selectedPerfRowIdAtom);
    const setSelectedPerfRowId = useSetAtom(selectedPerfRowIdAtom);

    const { sortTableFields, changeSorting, sortingColumn, sortDirection } = useSortTable(null);
    const opIdsMap = useOpToPerfIdFiltered();
    const { data: operationsSummary } = useOperationsSummary();
    const { data: npeManifest, error: npeManifestError } = useGetNPEManifest();
    const navigate = useNavigate();

    const tableFields = useMemo<TypedPerfTableRow[]>(() => {
        if (!data) {
            return [];
        }

        // Still some awkward casting here
        return [...sortTableFields(data as [])];
    }, [data, sortTableFields]);

    const comparisonDataTableFields = useMemo<TypedPerfTableRow[][]>(
        () =>
            comparisonData?.map((dataset) => {
                const parsedData = dataset;

                // Still some awkward casting here
                return [...sortTableFields(parsedData as [])];
            }) || [],
        [comparisonData, sortTableFields],
    );

    // L1 pressure is a per-TTNN-op snapshot, so it renders only on the first device-op row of each
    // op. Derive that "first" row from the current display order (`tableFields`, post-sort) rather
    // than source order, so the value always lands on the topmost visible row of its op group —
    // matching what the user sees under any sort instead of an arbitrary execution-order row.
    const firstRowOfOpRun = useMemo<Set<TypedPerfTableRow>>(() => {
        const seenOps = new Set<number>();
        const firstRows = new Set<TypedPerfTableRow>();

        for (const row of tableFields) {
            if (row.op !== undefined && !seenOps.has(row.op)) {
                seenOps.add(row.op);
                firstRows.add(row);
            }
        }

        return firstRows;
    }, [tableFields]);

    // Unsorted on purpose: eligibility is order-independent, so a sort click leaves the columns alone
    const rowsForColumnEligibility = useMemo(
        () => [...(data ?? EMPTY_ROWS), ...(comparisonData ?? EMPTY_COMPARISON_DATA).flat()],
        [data, comparisonData],
    );

    const eligibleColumns = useMemo(
        () =>
            getEligiblePerfColumns({
                hasOpIds: opIdsMap.length > 0,
                hasL1PressureData,
                hiliteHighDispatch,
                hasNpe: Boolean(npeManifest && npeManifest.length > 0),
                hasSubDeviceIds: rowsForColumnEligibility.some(
                    (row) => row.sub_device_id != null && row.sub_device_id !== '',
                ),
                hasMultipleAvailableCoreBudgets:
                    new Set(rowsForColumnEligibility.map((row) => row.available_cores).filter((value) => value != null))
                        .size > 1,
            }),
        [opIdsMap.length, hasL1PressureData, hiliteHighDispatch, npeManifest, rowsForColumnEligibility],
    );

    const visibleColumns = useMemo(
        () => getVisiblePerfColumns(eligibleColumns, hiddenColumns),
        [eligibleColumns, hiddenColumns],
    );

    const footerColumns = useMemo(() => getFooterColumns(visibleColumns), [visibleColumns]);

    const footerTotals = useMemo(() => computeFooterTotals(data, hideHostOps), [data, hideHostOps]);

    const isReportsSynced = opIdsMap.length > 0;
    const isPrimaryActiveReport = activeReportComparisonIndex === null;
    const activeReportRows = useMemo<TypedPerfTableRow[]>(
        () => (isPrimaryActiveReport ? tableFields : (comparisonDataTableFields[activeReportComparisonIndex] ?? [])),
        [isPrimaryActiveReport, tableFields, comparisonDataTableFields, activeReportComparisonIndex],
    );
    const canShowTensorDrawer = isReportsSynced && activeReportRows.length > 0;

    useEffect(() => {
        if (!canShowTensorDrawer) {
            setSelectedPerfRowId(null);
            return;
        }

        // Drop the selection when filters/range no longer include the row
        if (selectedPerfRowId !== null && !activeReportRows.some((row) => row.id === selectedPerfRowId)) {
            setSelectedPerfRowId(null);
        }
    }, [canShowTensorDrawer, activeReportRows, selectedPerfRowId, setSelectedPerfRowId]);

    const operationsById = useMemo(
        () => (operationsSummary ? getOperationsById(operationsSummary) : null),
        [operationsSummary],
    );

    const npeFileByGlobalCallCount = useMemo(() => getNpeFileByGlobalCallCount(npeManifest), [npeManifest]);

    const handleOpenNpeTimeline = useCallback((file: string) => navigate(`${ROUTES.NPE}/${file}`), [navigate]);

    // A new layout remounts the body, and with it the virtualiser, so no height measured under another
    // layout outlives it. See PerfTableBody.
    const tableBodyLayoutKey = useMemo(
        () =>
            [comparisonDataTableFields.length, provideMatmulAdvice, ...visibleColumns.map((column) => column.key)].join(
                ',',
            ),
        [comparisonDataTableFields.length, provideMatmulAdvice, visibleColumns],
    );

    const sizing = usePerfTableColumnSizing({
        rows: data ?? EMPTY_ROWS,
        comparisonRows: comparisonData ?? EMPTY_COMPARISON_DATA,
        visibleColumns,
        hasOperations: operationsById !== null,
        npeFileByGlobalCallCount,
        isPrimaryActiveReport,
        activeReportComparisonIndex,
        provideMatmulAdvice,
    });

    const renderTable = () => {
        if (isLoading) {
            return (
                <PerfTableSkeleton
                    headers={visibleColumns.map((column) => column.name)}
                    hasLeadingColumn
                />
            );
        }

        if (!data?.length) {
            return (
                <p>
                    <em>No data to display</em>
                </p>
            );
        }

        return (
            <table className='perf-table monospace'>
                <thead className='table-header'>
                    <tr>
                        <th
                            className='cell-header'
                            aria-label='Tensor details'
                        />
                        {visibleColumns.map((h) => {
                            const targetSortDirection =
                                // eslint-disable-next-line no-nested-ternary
                                sortingColumn === h.key
                                    ? sortDirection === SortingDirection.ASC
                                        ? SortingDirection.DESC
                                        : SortingDirection.ASC
                                    : sortDirection;

                            return (
                                <th
                                    key={h.key}
                                    className='cell-header'
                                >
                                    {h.sortable ? (
                                        <Button
                                            onClick={() => changeSorting(h.key)(targetSortDirection)}
                                            variant={ButtonVariant.MINIMAL}
                                            size={Size.SMALL}
                                        >
                                            <span className='header-label'>{h.name}</span>

                                            {sortingColumn === h.key ? (
                                                <Icon
                                                    className={classNames(
                                                        {
                                                            'is-active': sortingColumn === h.key,
                                                        },
                                                        'sort-icon',
                                                    )}
                                                    icon={
                                                        sortDirection === SortingDirection.ASC
                                                            ? IconNames.CARET_UP
                                                            : IconNames.CARET_DOWN
                                                    }
                                                />
                                            ) : (
                                                <Icon
                                                    className={classNames('sort-icon')}
                                                    icon={IconNames.CARET_DOWN}
                                                />
                                            )}
                                        </Button>
                                    ) : (
                                        <span className='header-label no-button'>{h.name}</span>
                                    )}

                                    {/* TODO: May want this in the near future */}
                                    {/* {h?.filterable && (
                                                <div className='column-filter'>
                                                    <InputGroup
                                                        asyncControl
                                                        size={Size.SMALL}
                                                        onValueChange={(value) => updateColumnFilter(h.key, value)}
                                                        placeholder='Filter...'
                                                        value={filters?.[h.key]}
                                                    />
                                                </div>
                                            )} */}
                                </th>
                            );
                        })}
                    </tr>
                </thead>

                <PerfTableBody
                    key={tableBodyLayoutKey}
                    rows={tableFields}
                    comparisonDatasets={comparisonDataTableFields}
                    sizing={sizing}
                    visibleColumns={visibleColumns}
                    filters={filters}
                    operationsById={operationsById}
                    npeFileByGlobalCallCount={npeFileByGlobalCallCount}
                    firstRowOfOpRun={firstRowOfOpRun}
                    selectedPerfRowId={selectedPerfRowId}
                    isPrimaryActiveReport={isPrimaryActiveReport}
                    activeReportComparisonIndex={activeReportComparisonIndex}
                    isReportsSynced={isReportsSynced}
                    provideMatmulAdvice={provideMatmulAdvice}
                    onSelectRow={setSelectedPerfRowId}
                    onOpenNpeTimeline={handleOpenNpeTimeline}
                />

                <tfoot className='table-footer'>
                    <tr>
                        <td />
                        {footerColumns.length > 0 &&
                            data?.length > 0 &&
                            footerColumns
                                .filter((header) => header?.footerSpan !== 0)
                                .map((header) => (
                                    <td
                                        key={header.key}
                                        className={classNames({
                                            'pre-wrap': header.key === ColumnKeys.OpCode,
                                        })}
                                        colSpan={header.footerSpan ?? undefined}
                                    >
                                        {footerTotals[header.key] ?? ''}
                                    </td>
                                ))}
                    </tr>
                </tfoot>
            </table>
        );
    };

    return (
        <>
            {npeManifestError && (
                <div className='error-message'>
                    <Icon
                        icon={IconNames.ERROR}
                        size={20}
                        intent={Intent.WARNING}
                    />
                    <p>Invalid NPE manifest: {npeManifestError.message}</p>
                </div>
            )}

            <PerfDeviceArchitecture
                reportFolderName={reportFolderName}
                maxCores={maxCores}
            />

            {mergeDevices && <PerfMultiDeviceNotice />}

            <PerfTableToolbar eligibleColumns={eligibleColumns} />

            {renderTable()}

            {canShowTensorDrawer && <PerfTensorDrawer rows={activeReportRows} />}
        </>
    );
};

const computeFooterTotals = (data: TypedPerfTableRow[], hideHostOps: boolean): Partial<Record<ColumnKeys, string>> => {
    const rows = data ?? [];
    let deviceTimeSum = 0;
    let opToOpGapSum = 0;
    let hostOpsCount = 0;
    let nonUniqueOpsCount = 0;
    let cacheHits = 0;

    for (const row of rows) {
        deviceTimeSum += row.device_time || 0;
        opToOpGapSum += row.op_to_op_gap || 0;

        if (isHostOp(row.bound)) {
            hostOpsCount++;
        }

        if (!row.isFirstHashOccurrence) {
            nonUniqueOpsCount++;

            if (row.cache_hit) {
                cacheHits++;
            }
        }
    }

    const deviceOpsCount = rows.length - hostOpsCount;
    const cacheHitPercent = nonUniqueOpsCount > 0 ? (cacheHits / nonUniqueOpsCount) * 100 : 0;

    return {
        [ColumnKeys.TotalPercent]: '100%',
        [ColumnKeys.DeviceTime]: `${formatSize(deviceTimeSum, 2)} µs`,
        [ColumnKeys.OpCode]: hideHostOps
            ? `${deviceOpsCount} device ops`
            : `${rows.length} ops\n(${deviceOpsCount} device ops + ${hostOpsCount} host ops)`,
        [ColumnKeys.OpToOpGap]: `${formatSize(opToOpGapSum, 2)} µs`,
        [ColumnKeys.CacheHit]: `${formatPercentage(cacheHitPercent).toString()} expected cache hits`,
    };
};

export default PerformanceTable;
