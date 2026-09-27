// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { Button, ButtonVariant, Intent, Size, Tooltip } from '@blueprintjs/core';
import { IconNames } from '@blueprintjs/icons';
import classNames from 'classnames';
import { memo } from 'react';
import { OpType, PATTERN_COUNT } from '../../definitions/Performance';
import { ColumnDefinition, ColumnKeys, PerfTableFilters, comparisonKeys } from '../../definitions/PerfTable';
import { TEST_IDS } from '../../definitions/TestIds';
import isValidNumber from '../../functions/isValidNumber';
import { OperationsById, formatCell, getPerfCellClassName, hasMatmulAdvice } from '../../functions/perfFunctions';
import { TypedPerfTableRow } from '../../model/PerfTable';

interface TensorDrawerStatus {
    canOpen: boolean;
    reason: string;
}

const getTensorDrawerStatus = (row: TypedPerfTableRow, isReportsSynced: boolean): TensorDrawerStatus => {
    if (!isReportsSynced) {
        return { canOpen: false, reason: 'Load a synced profiler report to see tensor details' };
    }

    if (row.raw_op_code.includes('MISSING')) {
        return { canOpen: false, reason: 'No tensor data for missing operations' };
    }

    if (!isValidNumber(row.op)) {
        return { canOpen: false, reason: 'No linked profiler operation for this row' };
    }

    return { canOpen: true, reason: 'View input/output tensor details' };
};

interface PerfTensorDrawerTriggerProps {
    row: TypedPerfTableRow;
    isReportsSynced: boolean;
    onSelectRow: (rowId: number | null) => void;
}

function PerfTensorDrawerTrigger({ row, isReportsSynced, onSelectRow }: PerfTensorDrawerTriggerProps) {
    if (row.op_type === OpType.SIGNPOST) {
        return null;
    }

    const status = getTensorDrawerStatus(row, isReportsSynced);

    return (
        <Tooltip content={status.reason}>
            {/* span wrapper lets the Tooltip attach to disabled buttons (Blueprint quirk) */}
            <span className='perf-tensor-trigger-wrapper'>
                <Button
                    className='perf-tensor-trigger'
                    disabled={!status.canOpen}
                    icon={IconNames.FLOW_LINEAR}
                    intent={Intent.PRIMARY}
                    variant={ButtonVariant.MINIMAL}
                    size={Size.SMALL}
                    aria-label={`View tensor details for ${row.raw_op_code}`}
                    data-testid={TEST_IDS.PERF_TENSOR_DRAWER_OPEN_BUTTON}
                    onClick={() => onSelectRow(row.id)}
                />
            </span>
        </Tooltip>
    );
}

interface PerfNpeLinkProps {
    row: TypedPerfTableRow;
    npeFileByGlobalCallCount: ReadonlyMap<number, string>;
    onOpenNpeTimeline: (file: string) => void;
}

function PerfNpeLink({ row, npeFileByGlobalCallCount, onOpenNpeTimeline }: PerfNpeLinkProps) {
    // NPE uses 0 as its "no value", so it is mapped out of the lookup rather than matched
    const globalCallCount = parseInt(String(row[ColumnKeys.GlobalCallCount]), 10) || -1;
    const npeFile = npeFileByGlobalCallCount.get(globalCallCount);

    if (npeFile === undefined) {
        return null;
    }

    return (
        <Tooltip content={`Launch NPE timeline for ${row.raw_op_code}`}>
            <Button
                icon={IconNames.Random}
                onClick={() => onOpenNpeTimeline(npeFile)}
                variant={ButtonVariant.MINIMAL}
                className='graph-button'
                aria-label={`Launch NPE timeline for ${row.raw_op_code}`}
                data-testid={TEST_IDS.PERF_NPE_LINK}
            />
        </Tooltip>
    );
}

/**
 * Props are a stability contract: the table re-renders this only when its own row, its selection or a
 * table-wide setting changes, so every value must be a primitive, a memoised value or a stable handler.
 */
interface PerfTableRowGroupProps {
    row: TypedPerfTableRow;
    /** Position in the sorted table; comparison sub-rows are index-aligned to it. */
    index: number;
    comparisonDatasets: TypedPerfTableRow[][];
    visibleColumns: ColumnDefinition[];
    filters: PerfTableFilters;
    operationsById: OperationsById | null;
    npeFileByGlobalCallCount: ReadonlyMap<number, string>;
    isFirstOfOpRun: boolean;
    /** Whether the group's active-report row (primary or comparison) is the selected row. */
    isSelected: boolean;
    isPrimaryActiveReport: boolean;
    activeReportComparisonIndex: number | null;
    isReportsSynced: boolean;
    provideMatmulAdvice: boolean;
    onSelectRow: (rowId: number | null) => void;
    onOpenNpeTimeline: (file: string) => void;
    measureElement: (element: HTMLTableSectionElement | null) => void;
}

/**
 * One table row with its comparison sub-rows and matmul advice, in its own `<tbody>` so the
 * virtualiser can measure the group as a unit.
 */
function PerfTableRowGroup({
    row,
    index,
    comparisonDatasets,
    visibleColumns,
    filters,
    operationsById,
    npeFileByGlobalCallCount,
    isFirstOfOpRun,
    isSelected,
    isPrimaryActiveReport,
    activeReportComparisonIndex,
    isReportsSynced,
    provideMatmulAdvice,
    onSelectRow,
    onOpenNpeTimeline,
    measureElement,
}: PerfTableRowGroupProps) {
    return (
        <tbody
            ref={measureElement}
            className='perf-table-row-group'
            data-index={index}
        >
            <tr
                className={classNames({
                    'missing-data': row.raw_op_code.includes('MISSING'),
                    'signpost-op': row.op_type === OpType.SIGNPOST,
                    'is-selected': isPrimaryActiveReport && isSelected,
                })}
            >
                <td className='cell'>
                    {isPrimaryActiveReport && (
                        <PerfTensorDrawerTrigger
                            row={row}
                            isReportsSynced={isReportsSynced}
                            onSelectRow={onSelectRow}
                        />
                    )}
                </td>
                {visibleColumns.map((column) => (
                    <td
                        key={column.key}
                        className={getPerfCellClassName(column)}
                    >
                        {column.key === ColumnKeys.GlobalCallCount ? (
                            <PerfNpeLink
                                row={row}
                                npeFileByGlobalCallCount={npeFileByGlobalCallCount}
                                onOpenNpeTimeline={onOpenNpeTimeline}
                            />
                        ) : (
                            formatCell(row, column, operationsById, filters?.[column.key], isFirstOfOpRun)
                        )}
                    </td>
                ))}
            </tr>

            {comparisonDatasets.map((dataset, datasetIndex) => {
                const subRow = dataset[index];
                const isActiveReportRow = datasetIndex === activeReportComparisonIndex;

                return (
                    <tr
                        key={`comparison-${datasetIndex}`}
                        className={classNames(
                            {
                                'missing-data': subRow?.raw_op_code?.includes('MISSING'),
                                'signpost-op': subRow?.op_type === OpType.SIGNPOST,
                                'is-selected': isActiveReportRow && isSelected,
                            },
                            'comparison-row',
                            `pattern-${datasetIndex >= PATTERN_COUNT ? datasetIndex - PATTERN_COUNT : datasetIndex}`,
                        )}
                    >
                        <td className='cell'>
                            {isActiveReportRow && subRow ? (
                                <PerfTensorDrawerTrigger
                                    row={subRow}
                                    isReportsSynced={isReportsSynced}
                                    onSelectRow={onSelectRow}
                                />
                            ) : null}
                        </td>
                        {visibleColumns.map((column) => (
                            <td
                                key={column.key}
                                className={getPerfCellClassName(column)}
                            >
                                {comparisonKeys.includes(column.key) &&
                                    subRow &&
                                    formatCell(subRow, column, operationsById, filters?.[column.key], true)}
                            </td>
                        ))}
                    </tr>
                );
            })}

            {provideMatmulAdvice && hasMatmulAdvice(row) && (
                <tr>
                    <td
                        colSpan={visibleColumns.length + 1}
                        className='cell advice'
                    >
                        <ul>
                            {row.advice.map((advice, adviceIndex) => (
                                <li key={`advice-${adviceIndex}`}>{advice}</li>
                            ))}
                        </ul>
                    </td>
                </tr>
            )}
        </tbody>
    );
}

export default memo(PerfTableRowGroup);
