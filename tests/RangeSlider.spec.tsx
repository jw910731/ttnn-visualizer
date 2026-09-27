// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import '@testing-library/jest-dom/vitest';
import { cleanup, render } from '@testing-library/react';
import { Mock, afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Range from '../src/components/RangeSlider';
import ROUTES from '../src/definitions/Routes';
import {
    useGetClusterDescription,
    useGetDeviceOperationListPerf,
    useOpToPerfIdFiltered,
    useOperationListRange,
    useOperationsSummary,
    usePerformanceRange,
    usePerformanceReport,
} from '../src/hooks/useAPI';
import { OperationSummary } from '../src/model/APIData';
import { PerfTableRow } from '../src/model/PerfTable';
import { selectedPerformanceRangeAtom } from '../src/store/app';
import { TestProviders } from './helpers/TestProviders';

vi.mock('../src/hooks/useAPI.tsx', () => ({
    useGetClusterDescription: vi.fn(),
    useGetDeviceOperationListPerf: vi.fn(),
    useOpToPerfIdFiltered: vi.fn(),
    useOperationListRange: vi.fn(),
    useOperationsSummary: vi.fn(),
    // The slider is mounted on every route, and the full list is hundreds of MB on a large
    // report; its labels need only ids and names, which the summary carries.
    useOperationsList: () => {
        throw new Error('Range must read useOperationsSummary, not the full operations list');
    },
    usePerformanceRange: vi.fn(),
    usePerformanceReport: vi.fn(),
}));

const perfRow = (id: string, rawOpCode: string) => ({ id, raw_op_code: rawOpCode }) as PerfTableRow;

const operationSummary = (id: number, name: string): OperationSummary => ({
    id,
    name,
    operation_file_identifier: '',
    device_operation_start_names: [],
    device_operation_end_names: [],
});

// The id the slider hands the label renderer is a number; the report's ids are strings.
const REPORT = [perfRow('2', 'Matmul'), perfRow('3', 'Softmax'), perfRow('3', 'Duplicate'), perfRow('4', 'Reduce')];

const LARGE_REPORT_ROW_COUNT = 200;
const HANDLE_LABEL_READ_ALLOWANCE = 20;

const handleLabels = (container: HTMLElement) =>
    Array.from(container.querySelectorAll('.bp6-slider-handle .bp6-slider-label')).map((label) => label.textContent);

// The slider syncs its selection to the report's span on mount, so the span is where the
// handles land.
const renderRange = (performanceRange: [number, number]) => {
    (usePerformanceRange as Mock).mockReturnValue(performanceRange);

    return render(
        <TestProviders
            initialEntries={[ROUTES.PERFORMANCE]}
            initialAtomValues={[[selectedPerformanceRangeAtom, performanceRange]]}
        >
            <Range />
        </TestProviders>,
    );
};

afterEach(cleanup);

beforeEach(() => {
    (useOperationsSummary as Mock).mockReturnValue({ data: undefined });
    (usePerformanceReport as Mock).mockReturnValue({
        data: { report: REPORT, stacked_report: [], signposts: [] },
        error: null,
    });
    (useGetClusterDescription as Mock).mockReturnValue({ error: null });
    (useGetDeviceOperationListPerf as Mock).mockReturnValue([]);
    (useOpToPerfIdFiltered as Mock).mockReturnValue([]);
    (useOperationListRange as Mock).mockReturnValue(null);
});

describe('Range performance labels', () => {
    it('names the report row each handle sits on', () => {
        const { container } = renderRange([2, 4]);

        expect(handleLabels(container)).toEqual(['2\xA0Matmul', '4\xA0Reduce']);
    });

    it('names the first row when the report repeats an id', () => {
        const { container } = renderRange([3, 4]);

        expect(handleLabels(container)).toEqual(['3\xA0Softmax', '4\xA0Reduce']);
    });

    it('falls back to the bare id where no row carries it', () => {
        const { container } = renderRange([1, 9]);

        expect(handleLabels(container)).toEqual(['1', '9']);
    });

    // The slider labels every tick and both handles on each render. Finding each label's row by
    // scanning the report read every row's id once per label, per render.
    it('reads each row id once per report rather than once per label', () => {
        let idReads = 0;
        const report = Array.from({ length: LARGE_REPORT_ROW_COUNT }, (_, index) =>
            Object.defineProperty({ raw_op_code: 'Matmul' }, 'id', {
                get: () => {
                    idReads += 1;
                    return String(index + 1);
                },
            }),
        ) as PerfTableRow[];
        (usePerformanceReport as Mock).mockReturnValue({
            data: { report, stacked_report: [], signposts: [] },
            error: null,
        });

        const { container } = renderRange([1, LARGE_REPORT_ROW_COUNT]);

        expect(handleLabels(container)).toEqual(['1\xA0Matmul', `${LARGE_REPORT_ROW_COUNT}\xA0Matmul`]);
        // One read per row to index them, plus the handful the rendered handle labels print
        expect(idReads).toBeLessThan(LARGE_REPORT_ROW_COUNT + HANDLE_LABEL_READ_ALLOWANCE);
    });
});

describe('Range operation labels', () => {
    // Off the performance route, where the operation slider is the one the user drives.
    const renderOperationRange = (operationRange: [number, number]) => {
        (useOperationListRange as Mock).mockReturnValue(operationRange);
        (usePerformanceRange as Mock).mockReturnValue(null);
        (usePerformanceReport as Mock).mockReturnValue({ data: undefined, error: null });

        return render(
            <TestProviders initialEntries={[ROUTES.OPERATIONS]}>
                <Range />
            </TestProviders>,
        );
    };

    it('names the operation each handle sits on from the operations summary', () => {
        (useOperationsSummary as Mock).mockReturnValue({
            data: [
                operationSummary(2, 'ttnn.matmul'),
                operationSummary(3, 'ttnn.softmax'),
                operationSummary(4, 'ttnn.sum'),
            ],
        });

        const { container } = renderOperationRange([2, 4]);

        expect(handleLabels(container)).toEqual(['2\xA0ttnn.matmul', '4\xA0ttnn.sum']);
    });

    it('names the first operation when the summary repeats an id', () => {
        (useOperationsSummary as Mock).mockReturnValue({
            data: [
                operationSummary(2, 'ttnn.matmul'),
                operationSummary(2, 'ttnn.shadowed'),
                operationSummary(4, 'ttnn.sum'),
            ],
        });

        const { container } = renderOperationRange([2, 4]);

        expect(handleLabels(container)).toEqual(['2\xA0ttnn.matmul', '4\xA0ttnn.sum']);
    });

    it('falls back to the bare id while the summary is loading', () => {
        const { container } = renderOperationRange([2, 4]);

        expect(handleLabels(container)).toEqual(['2', '4']);
    });
});
