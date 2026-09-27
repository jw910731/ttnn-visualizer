// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { cleanup, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import { Provider, createStore } from 'jotai';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useOperationListRange, useOperationsSummary } from '../src/hooks/useAPI';
import { activeProfilerReportAtom } from '../src/store/app';
import axiosInstance from '../src/libs/axiosInstance';
import Endpoints from '../src/definitions/Endpoints';
import { OperationSummary } from '../src/model/APIData';
import { ReportFolder } from '../src/definitions/Reports';

vi.mock('../src/libs/axiosInstance', () => ({
    default: {
        get: vi.fn(),
    },
    getOrCreateInstanceId: () => 'test-instance',
}));

const REPORT_A = { path: 'report-a', reportName: 'report-a' } as ReportFolder;
const REPORT_B = { path: 'report-b', reportName: 'report-b' } as ReportFolder;

const summary = (id: number, name: string): OperationSummary => ({
    id,
    name,
    operation_file_identifier: 'model.py:1',
    device_operation_start_names: [],
    device_operation_end_names: [],
});

const SUMMARY_BY_REPORT: Record<string, OperationSummary[]> = {
    [REPORT_A.path]: [summary(2, 'ttnn.matmul'), summary(5, 'ttnn.add'), summary(9, 'ttnn.relu')],
    [REPORT_B.path]: [summary(1, 'ttnn.softmax')],
};

const getSummaryRequests = () =>
    vi.mocked(axiosInstance.get).mock.calls.filter(([url]) => url === Endpoints.OPERATIONS_SUMMARY);

const renderWithStore = <T,>(hook: () => T, store: ReturnType<typeof createStore>) => {
    const queryClient = new QueryClient();

    return renderHook(hook, {
        wrapper: ({ children }: { children: ReactNode }) => (
            <QueryClientProvider client={queryClient}>
                <Provider store={store}>{children}</Provider>
            </QueryClientProvider>
        ),
    });
};

let store: ReturnType<typeof createStore>;

beforeEach(() => {
    vi.clearAllMocks();
    store = createStore();
    vi.mocked(axiosInstance.get).mockImplementation((url: string) => {
        if (url === Endpoints.OPERATIONS_SUMMARY) {
            const report = store.get(activeProfilerReportAtom);

            return Promise.resolve({ data: SUMMARY_BY_REPORT[report?.path ?? ''] ?? [] });
        }

        return Promise.reject(new Error(`unexpected request to ${url}`));
    });
});

afterEach(cleanup);

describe('useOperationsSummary', () => {
    it('reads the summary endpoint for the active report', async () => {
        store.set(activeProfilerReportAtom, REPORT_A);
        const { result } = renderWithStore(() => useOperationsSummary(), store);

        await waitFor(() => expect(result.current.data).toEqual(SUMMARY_BY_REPORT[REPORT_A.path]));

        expect(getSummaryRequests()).toHaveLength(1);
    });

    it('asks for nothing while no memory report is loaded', async () => {
        const { result } = renderWithStore(() => useOperationsSummary(), store);

        // Let any request the hook would make go out before asserting there was none
        await new Promise((resolve) => {
            setTimeout(resolve, 0);
        });

        expect(result.current.data).toBeUndefined();
        expect(result.current.fetchStatus).toBe('idle');
        expect(getSummaryRequests()).toHaveLength(0);
    });

    // Operation ids restart in every report, so a key without the report would serve the
    // previous report's names for the next one's ids (#1674).
    it('keys the summary by report, fetching again when another report is loaded', async () => {
        store.set(activeProfilerReportAtom, REPORT_A);
        const { result } = renderWithStore(() => useOperationsSummary(), store);

        await waitFor(() => expect(result.current.data).toEqual(SUMMARY_BY_REPORT[REPORT_A.path]));

        store.set(activeProfilerReportAtom, REPORT_B);

        await waitFor(() => expect(result.current.data).toEqual(SUMMARY_BY_REPORT[REPORT_B.path]));
        expect(getSummaryRequests()).toHaveLength(2);
    });

    // What axios hands back when the body is not JSON. Succeeding with it would pass a string to
    // the report match, which throws while the footer renders on every route.
    it('fails, rather than succeeding with a string, when the body is not a list', async () => {
        vi.mocked(axiosInstance.get).mockResolvedValue({ data: '<html>not json</html>' });
        store.set(activeProfilerReportAtom, REPORT_A);
        const { result } = renderWithStore(() => useOperationsSummary(), store);

        await waitFor(() => expect(result.current.isError).toBe(true));

        expect(result.current.data).toBeUndefined();
        expect(result.current.error?.code).toBe(AxiosError.ERR_BAD_RESPONSE);
    });

    it('does not retry a failed request, as the operations list does not', async () => {
        vi.mocked(axiosInstance.get).mockRejectedValue(new Error('Request failed with status code 500'));
        store.set(activeProfilerReportAtom, REPORT_A);
        const { result } = renderWithStore(() => useOperationsSummary(), store);

        await waitFor(() => expect(result.current.isError).toBe(true));

        expect(getSummaryRequests()).toHaveLength(1);
    });
});

describe('useOperationListRange', () => {
    it('spans the first and last operation id of the summary', async () => {
        store.set(activeProfilerReportAtom, REPORT_A);
        const { result } = renderWithStore(() => useOperationListRange(), store);

        await waitFor(() => expect(result.current).toEqual([2, 9]));

        expect(vi.mocked(axiosInstance.get).mock.calls.map(([url]) => url)).toEqual([Endpoints.OPERATIONS_SUMMARY]);
    });

    it('has no range while no memory report is loaded', () => {
        const { result } = renderWithStore(() => useOperationListRange(), store);

        expect(result.current).toBeNull();
    });
});
