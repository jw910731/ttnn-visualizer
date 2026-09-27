// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { Button, Drawer, DrawerSize, Intent } from '@blueprintjs/core';
import { IconNames } from '@blueprintjs/icons';
import { useAtom } from 'jotai';
import { useMemo } from 'react';
import { useNavigate } from 'react-router';
import 'styles/components/PerfTensorDrawer.scss';
import { TypedPerfTableRow } from '../../model/PerfTable';
import ROUTES from '../../definitions/Routes';
import { TEST_IDS } from '../../definitions/TestIds';
import { StackTraceLanguage } from '../../definitions/StackTrace';
import isValidNumber from '../../functions/isValidNumber';
import getResponseError from '../../functions/getResponseError';
import { useOperationsList } from '../../hooks/useAPI';
import { OperationDescription } from '../../model/APIData';
import { selectedPerfRowIdAtom } from '../../store/app';
import PerfTensorPanel from './PerfTensorPanel';
import LoadingSpinner from '../LoadingSpinner';
import SourceFileButton from '../operation-details/SourceFileButton';
import { extractOperationSourceData } from '../../functions/stackTraceSource';

const NO_OPERATIONS: OperationDescription[] = [];
const OPERATIONS_ERROR_FALLBACK = 'Failed to load profiler operations';

interface PerfTensorDrawerProps {
    rows: TypedPerfTableRow[];
}

/**
 * The shell reads no operations itself: only the title and details below subscribe to the
 * full operations list, and they render only while a row is selected. That list carries
 * every operation's graph and tensors, and nothing else on the performance route needs it,
 * so the route does not fetch it until someone opens tensor details.
 */
function PerfTensorDrawer({ rows }: PerfTensorDrawerProps) {
    const [selectedPerfRowId, setSelectedPerfRowId] = useAtom(selectedPerfRowIdAtom);

    const selectedRow = rows.find((row) => row.id === selectedPerfRowId) ?? null;

    const handleClose = () => {
        setSelectedPerfRowId(null);
    };

    return (
        <Drawer
            isOpen={selectedRow !== null}
            onClose={handleClose}
            title={selectedRow ? <PerfTensorDrawerTitle row={selectedRow} /> : 'Tensor details'}
            size={DrawerSize.SMALL}
            className='perf-tensor-drawer'
        >
            <div
                className='perf-tensor-drawer-content'
                data-testid={TEST_IDS.PERF_TENSOR_DRAWER}
            >
                {selectedRow ? <PerfTensorDrawerDetails row={selectedRow} /> : <NoLinkedOperation />}
            </div>
        </Drawer>
    );
}

const useLinkedOperation = (row: TypedPerfTableRow) => {
    const { data: operations = NO_OPERATIONS, isLoading, error } = useOperationsList();
    const operationId = row.op;

    const operation = useMemo(
        () =>
            isValidNumber(operationId) ? (operations.find((candidate) => candidate.id === operationId) ?? null) : null,
        [operations, operationId],
    );

    return { operation, operations, isLoading, error };
};

interface PerfTensorDrawerRowProps {
    row: TypedPerfTableRow;
}

function PerfTensorDrawerTitle({ row }: PerfTensorDrawerRowProps) {
    const { operation } = useLinkedOperation(row);

    return (
        <span className='perf-tensor-drawer-title'>
            {operation?.id} {operation?.name}
        </span>
    );
}

function PerfTensorDrawerDetails({ row }: PerfTensorDrawerRowProps) {
    const { operation, operations, isLoading, error } = useLinkedOperation(row);
    const navigate = useNavigate();

    // The list is fetched on first open rather than with the route, so without this the
    // drawer would claim there is no linked operation for as long as the fetch takes.
    if (isLoading) {
        return <LoadingSpinner />;
    }

    if (error) {
        return (
            <p className='perf-tensor-drawer-empty'>
                <em>{getResponseError(error, OPERATIONS_ERROR_FALLBACK)}</em>
            </p>
        );
    }

    if (!operation) {
        return <NoLinkedOperation />;
    }

    const operationSourceData = extractOperationSourceData(operation);

    return (
        <>
            <p className='perf-tensor-drawer-op-link'>
                <Button
                    className='navigate-button'
                    endIcon={IconNames.SEGMENTED_CONTROL}
                    intent={Intent.PRIMARY}
                    onClick={() => navigate(`${ROUTES.OPERATIONS}/${operation.id}`)}
                >
                    Memory Details
                </Button>
                {operationSourceData && (
                    <SourceFileButton
                        filePath={operationSourceData.filePath}
                        sourceFileId={operation.stack_trace_source_file_id ?? null}
                        lineNumber={operationSourceData.lineNumber}
                        language={StackTraceLanguage.PYTHON}
                        testId={TEST_IDS.SHOW_OPERATION_SOURCE_BUTTON}
                        ariaLabel={`View source for operation ${operation.id} ${operation.name}`}
                        eagerProbe
                    />
                )}
            </p>

            <PerfTensorPanel
                operation={operation}
                operations={operations}
            />
        </>
    );
}

function NoLinkedOperation() {
    return (
        <p className='perf-tensor-drawer-empty'>
            <em>No linked profiler operation for this row.</em>
        </p>
    );
}

export default PerfTensorDrawer;
