// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { cleanup, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useOperationDetails, useOperationsList } from '../src/hooks/useAPI';
import { activeProfilerReportAtom } from '../src/store/app';
import axiosInstance from '../src/libs/axiosInstance';
import Endpoints from '../src/definitions/Endpoints';
import { NodeType } from '../src/model/APIData';
import { AtomProvider } from './helpers/atomProvider';

vi.mock('../src/libs/axiosInstance', () => ({
    default: {
        get: vi.fn(),
    },
    getOrCreateInstanceId: () => 'test-instance',
}));

const ACTIVE_REPORT = { path: 'memory-report', reportName: 'memory-report' };
const OPERATION_ID = 1;

// Older report databases name a node by `counter` alone; newer ones carry both, and
// deliberately different values here so a case can tell which one `id` came from.
const LEGACY_NODE_COUNTER = 7;
const CURRENT_NODE_ID = 3;
const CURRENT_NODE_COUNTER = 30;

// Built per response, since the normalisation writes into the nodes it is handed
const capturedGraph = () => [
    { counter: LEGACY_NODE_COUNTER, node_type: NodeType.function_start, params: { name: 'MatmulDeviceOperation' } },
    {
        id: CURRENT_NODE_ID,
        counter: CURRENT_NODE_COUNTER,
        node_type: NodeType.function_end,
        params: { name: 'MatmulDeviceOperation' },
    },
];

const operation = () => ({
    id: OPERATION_ID,
    name: 'ttnn.matmul',
    stack_trace: '',
    inputs: [],
    outputs: [],
    arguments: [],
    device_operations: capturedGraph(),
});

const renderWithReport = <T,>(hook: () => T) => {
    const queryClient = new QueryClient();

    return renderHook(hook, {
        wrapper: ({ children }: { children: ReactNode }) => (
            <QueryClientProvider client={queryClient}>
                <AtomProvider initialValues={[[activeProfilerReportAtom, ACTIVE_REPORT]]}>{children}</AtomProvider>
            </QueryClientProvider>
        ),
    });
};

beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(axiosInstance.get).mockImplementation((url: string) => {
        if (url === `${Endpoints.OPERATIONS_LIST}/${OPERATION_ID}`) {
            return Promise.resolve({ data: { ...operation(), buffers: [], l1_sizes: [] } });
        }

        if (url === Endpoints.OPERATIONS_LIST) {
            return Promise.resolve({ data: [operation()] });
        }

        return Promise.resolve({ data: [] });
    });
});

afterEach(cleanup);

describe('device operation node ids', () => {
    it('names a node that only has a counter by it, and leaves an existing id alone', async () => {
        const { result } = renderWithReport(() => useOperationsList());

        await waitFor(() => expect(result.current.data).toHaveLength(1));

        const [legacyNode, currentNode] = result.current.data![0].device_operations;

        expect(legacyNode.id).toBe(LEGACY_NODE_COUNTER);
        expect(currentNode.id).toBe(CURRENT_NODE_ID);
    });

    // Deleting it put every node of a large report into V8's dictionary mode: far more
    // memory per node and several times slower walks over them afterwards.
    it('leaves the counter in place rather than deleting it', async () => {
        const { result } = renderWithReport(() => useOperationsList());

        await waitFor(() => expect(result.current.data).toHaveLength(1));

        for (const node of result.current.data![0].device_operations) {
            expect(node).toHaveProperty('counter');
        }
    });

    it('normalises the nodes of a single operation’s details the same way', async () => {
        const { result } = renderWithReport(() => useOperationDetails(OPERATION_ID));

        await waitFor(() => expect(result.current.operationDetails.data?.device_operations).toHaveLength(2));

        const [legacyNode, currentNode] = result.current.operationDetails.data!.device_operations;

        expect(legacyNode).toMatchObject({ id: LEGACY_NODE_COUNTER, counter: LEGACY_NODE_COUNTER });
        expect(currentNode).toMatchObject({ id: CURRENT_NODE_ID, counter: CURRENT_NODE_COUNTER });
    });
});

describe('operations list', () => {
    // The device-operation graph resolves its frames on expansion; attaching them to every
    // operation on load held tens of MiB on a large report for a view that may never open.
    it('does not resolve device-operation frames on load', async () => {
        const { result } = renderWithReport(() => useOperationsList());

        await waitFor(() => expect(result.current.data).toHaveLength(1));

        expect(result.current.data![0]).not.toHaveProperty('processedConnections');
        expect(result.current.data![0].deviceOperationNameList).toEqual(['MatmulDeviceOperation']);
    });
});
