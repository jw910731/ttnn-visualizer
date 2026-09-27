// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

/**
 * The full operations list is every operation's captured graph and tensors: 280 MB and ~10 s of
 * backend time on an 18,740-operation report. The performance route and the footer mounted with it
 * need only ids, names, file identifiers and device-operation names, so they read the summary and
 * leave the full list to the tensor drawer, which asks for it only once opened.
 */

import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Performance from '../src/routes/Performance';
import FooterInfobar from '../src/components/FooterInfobar';
import Endpoints from '../src/definitions/Endpoints';
import ROUTES from '../src/definitions/Routes';
import { TEST_IDS } from '../src/definitions/TestIds';
import { ReportLinkMatchResult } from '../src/definitions/ReportLinks';
import { useReportLinkMatch } from '../src/hooks/useReportLinkMatch';
import { OpType } from '../src/definitions/Performance';
import axiosInstance from '../src/libs/axiosInstance';
import { OperationSummary } from '../src/model/APIData';
import { PerfTableRow } from '../src/model/PerfTable';
import { activePerformanceReportAtom, activeProfilerReportAtom } from '../src/store/app';
import { TestProviders } from './helpers/TestProviders';

vi.mock('../src/libs/axiosInstance', () => ({
    default: {
        get: vi.fn(),
    },
    getOrCreateInstanceId: () => 'test-instance',
}));

vi.mock('../src/functions/getServerConfig', () => ({
    default: () => ({ SERVER_MODE: false, BASE_PATH: '/' }),
    isFlagEnabled: (value: unknown) => value === true || value === 'true' || value === '1',
}));

const PROFILER_REPORT = { path: 'memory-report', reportName: 'memory-report' };
const PERFORMANCE_REPORT = { path: 'performance-report', reportName: 'performance-report' };
const OPERATION_ID = 7;
const DEVICE_OPERATION_NAME = 'MatmulDeviceOperation';

const OPERATION_SUMMARY: OperationSummary = {
    id: OPERATION_ID,
    name: 'ttnn.matmul',
    operation_file_identifier: 'model.py:12',
    device_operation_start_names: [DEVICE_OPERATION_NAME],
    device_operation_end_names: [DEVICE_OPERATION_NAME],
};

const FULL_OPERATION = {
    id: OPERATION_ID,
    name: 'ttnn.matmul',
    duration: 1,
    stack_trace: 'File "/models/model.py", line 12, in forward',
    stack_trace_source_file_id: null,
    device_operations: [],
    arguments: [],
    inputs: [],
    outputs: [],
    error: null,
};

const PERF_ROW = {
    id: '1',
    global_call_count: 0,
    advice: [],
    total_percent: '100',
    bound: 'DRAM',
    op_code: DEVICE_OPERATION_NAME,
    raw_op_code: DEVICE_OPERATION_NAME,
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

const getOperationsListRequests = () =>
    vi.mocked(axiosInstance.get).mock.calls.filter(([url]) => url === Endpoints.OPERATIONS_LIST);

const getOperationsSummaryRequests = () =>
    vi.mocked(axiosInstance.get).mock.calls.filter(([url]) => url === Endpoints.OPERATIONS_SUMMARY);

const PerformanceRouteWithFooter = ({ children }: { children?: ReactNode }) => (
    <TestProviders
        initialEntries={[ROUTES.PERFORMANCE]}
        initialAtomValues={[
            [activeProfilerReportAtom, PROFILER_REPORT],
            [activePerformanceReportAtom, PERFORMANCE_REPORT],
        ]}
    >
        <Performance />
        <FooterInfobar />
        {children}
    </TestProviders>
);

const renderPerformanceRoute = () => render(<PerformanceRouteWithFooter />);

afterEach(cleanup);

beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(window, 'scrollTo').mockImplementation(() => {});
    vi.mocked(axiosInstance.get).mockImplementation((url: string) => {
        if (url === Endpoints.OPERATIONS_SUMMARY) {
            return Promise.resolve({ data: [OPERATION_SUMMARY] });
        }

        if (url === Endpoints.OPERATIONS_LIST) {
            return Promise.resolve({ data: [FULL_OPERATION] });
        }

        if (url === Endpoints.PERFORMANCE_RESULTS_REPORT) {
            // A fresh array per response, as the real endpoint's would be
            return Promise.resolve({ data: { report: [{ ...PERF_ROW }], stacked_report: [], signposts: [] } });
        }

        if (url === Endpoints.DEVICES) {
            return Promise.resolve({ data: [{ device_id: 0 }] });
        }

        return Promise.resolve({ data: null });
    });
});

describe('performance route operations fetch', () => {
    it('links the reports from the summary and asks for the full list only when tensor details open', async () => {
        renderPerformanceRoute();

        const trigger = await screen.findByTestId(TEST_IDS.PERF_TENSOR_DRAWER_OPEN_BUTTON);
        await waitFor(() => expect(trigger).toBeEnabled());

        expect(getOperationsSummaryRequests().length).toBeGreaterThan(0);
        expect(getOperationsListRequests()).toHaveLength(0);

        fireEvent.click(trigger);

        expect(await screen.findByRole('button', { name: 'Memory Details' })).toBeInTheDocument();
        expect(getOperationsListRequests()).toHaveLength(1);
    });

    // A proxy's error page or a cut-short body reaches the client as a string. The footer is on
    // every route, so it must not take the page down with it.
    it('keeps the route and footer up, with the link unavailable, when the summary is not JSON', async () => {
        vi.mocked(axiosInstance.get).mockImplementation((url: string) => {
            if (url === Endpoints.OPERATIONS_SUMMARY) {
                return Promise.resolve({ data: '<html>not json</html>' });
            }

            if (url === Endpoints.PERFORMANCE_RESULTS_REPORT) {
                return Promise.resolve({ data: { report: [{ ...PERF_ROW }], stacked_report: [], signposts: [] } });
            }

            if (url === Endpoints.DEVICES) {
                return Promise.resolve({ data: [{ device_id: 0 }] });
            }

            return Promise.resolve({ data: null });
        });

        const { result } = renderHook(() => useReportLinkMatch(), { wrapper: PerformanceRouteWithFooter });

        await waitFor(() => expect(result.current).toBe(ReportLinkMatchResult.UNAVAILABLE));
        expect(screen.getByText('Memory:')).toBeInTheDocument();
        expect(screen.getByText('Performance:')).toBeInTheDocument();
        expect(getOperationsListRequests()).toHaveLength(0);
    });
});
