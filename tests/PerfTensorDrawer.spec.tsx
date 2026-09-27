// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import '@testing-library/jest-dom/vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { Mock, afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import PerfTensorDrawer from '../src/components/performance/PerfTensorDrawer';
import { TypedPerfTableRow } from '../src/model/PerfTable';
import { OpType } from '../src/definitions/Performance';
import ROUTES from '../src/definitions/Routes';
import { TEST_IDS } from '../src/definitions/TestIds';
import { useOperationsList } from '../src/hooks/useAPI';
import { OperationDescription, Tensor } from '../src/model/APIData';
import { BufferType } from '../src/model/BufferType';
import { selectedPerfRowIdAtom } from '../src/store/app';
import { TestProviders } from './helpers/TestProviders';

vi.mock('../src/hooks/useAPI.tsx', () => ({
    useOperationsList: vi.fn(),
    useReportMetadata: vi.fn().mockReturnValue({ data: undefined }),
}));

vi.mock('../src/hooks/useRemote', () => ({
    default: () => ({
        readRemoteFile: vi.fn().mockResolvedValue({ data: '', error: null, resolvedPath: null }),
        isSourceFileAvailable: vi.fn().mockResolvedValue({ available: false, source: null }),
        persistentState: { selectedConnection: null },
    }),
}));

const tensor: Tensor = {
    id: 100,
    address: 4096,
    buffer_type: BufferType.L1,
    producers: [10],
    consumers: [12],
    producerNames: [],
    consumerNames: [],
    shape: 'Shape([1, 32, 32])',
    dtype: 'DataType::BFLOAT16',
    layout: 'Layout::TILE',
    memory_config: null,
    device_id: 0,
    comparison: null,
    io: 'input',
    size: 2048,
};

const matmulOp: OperationDescription = {
    id: 11,
    name: 'matmul_op',
    inputs: [tensor],
    outputs: [{ ...tensor, id: 101, io: 'output', producers: [11], consumers: [] }],
    stack_trace: 'File "/models/matmul.py", line 10, in forward\n    return ttnn.matmul(a, b)',
    stack_trace_source_file_id: 3,
    device_operations: [],
    operationFileIdentifier: 'matmul.cpp',
    error: null,
    duration: 1,
    arguments: [],
    deviceOperationNameList: ['Matmul'],
};

const makeRow = (
    overrides: Partial<TypedPerfTableRow> & Pick<TypedPerfTableRow, 'id' | 'raw_op_code'>,
): TypedPerfTableRow =>
    ({
        op_type: OpType.DEVICE_OP,
        op_code: overrides.raw_op_code,
        advice: [],
        bound: null,
        isFirstHashOccurrence: true,
        ...overrides,
    }) as unknown as TypedPerfTableRow;

const mappedRow = makeRow({ id: 1, raw_op_code: 'Matmul', op: 11 });
const unmappedRow = makeRow({ id: 2, raw_op_code: 'Reshape', op: undefined });

afterEach(cleanup);

beforeEach(() => {
    vi.clearAllMocks();
    (useOperationsList as Mock).mockReturnValue({ data: [matmulOp], isLoading: false, error: null });
});

describe('PerfTensorDrawer', () => {
    it('renders the operation link and tensor panel when the selected row maps to an op', () => {
        render(
            <TestProviders initialAtomValues={[[selectedPerfRowIdAtom, mappedRow.id]]}>
                <PerfTensorDrawer rows={[mappedRow]} />
            </TestProviders>,
        );

        expect(screen.getByTestId(TEST_IDS.PERF_TENSOR_DRAWER)).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: /11\s+matmul_op/ })).toBeInTheDocument();

        expect(screen.getByRole('button', { name: 'Memory Details' })).toBeInTheDocument();
        expect(screen.getByTestId(TEST_IDS.SHOW_OPERATION_SOURCE_BUTTON)).toBeInTheDocument();

        const opLink = screen.getByRole('link', { name: /11 matmul_op \(matmul\.cpp\)/i });
        expect(opLink).toHaveAttribute('href', `${ROUTES.OPERATIONS}/11`);

        expect(screen.getByText('Inputs')).toBeInTheDocument();
        expect(screen.getByText('Outputs')).toBeInTheDocument();
        expect(screen.getByText('100')).toBeInTheDocument();
        expect(screen.getByText('101')).toBeInTheDocument();
    });

    it('shows the empty-state message when the selected row has no matching operation', () => {
        render(
            <TestProviders initialAtomValues={[[selectedPerfRowIdAtom, unmappedRow.id]]}>
                <PerfTensorDrawer rows={[unmappedRow]} />
            </TestProviders>,
        );

        expect(screen.getByText('No linked profiler operation for this row.')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Memory Details' })).not.toBeInTheDocument();
        expect(screen.queryByRole('link', { name: /matmul_op/i })).not.toBeInTheDocument();
    });

    // The drawer is mounted whenever the table can show it, and the full list is every
    // operation's graph and tensors; subscribing while closed fetched it for the whole route.
    it('does not read the operations list while no row is selected', () => {
        render(
            <TestProviders initialAtomValues={[[selectedPerfRowIdAtom, null]]}>
                <PerfTensorDrawer rows={[mappedRow, unmappedRow]} />
            </TestProviders>,
        );

        expect(useOperationsList).not.toHaveBeenCalled();
    });

    it('shows a spinner rather than the empty state while the operations list loads', () => {
        (useOperationsList as Mock).mockReturnValue({ data: undefined, isLoading: true, error: null });

        render(
            <TestProviders initialAtomValues={[[selectedPerfRowIdAtom, mappedRow.id]]}>
                <PerfTensorDrawer rows={[mappedRow]} />
            </TestProviders>,
        );

        const drawer = screen.getByTestId(TEST_IDS.PERF_TENSOR_DRAWER);

        expect(drawer.querySelector('.loading-spinner')).toBeInTheDocument();
        expect(screen.queryByText('No linked profiler operation for this row.')).not.toBeInTheDocument();
    });

    it('says why when the operations list failed to load', () => {
        (useOperationsList as Mock).mockReturnValue({
            data: undefined,
            isLoading: false,
            error: new Error('Request failed with status code 500'),
        });

        render(
            <TestProviders initialAtomValues={[[selectedPerfRowIdAtom, mappedRow.id]]}>
                <PerfTensorDrawer rows={[mappedRow]} />
            </TestProviders>,
        );

        expect(screen.getByText('Request failed with status code 500')).toBeInTheDocument();
        expect(screen.queryByText('No linked profiler operation for this row.')).not.toBeInTheDocument();
    });
});
