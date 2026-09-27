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
    useOperationsList,
    usePerformanceRange,
    usePerformanceReport,
} from '../src/hooks/useAPI';
import { PerfTableRow } from '../src/model/PerfTable';
import { selectedPerformanceRangeAtom } from '../src/store/app';
import { TestProviders } from './helpers/TestProviders';

vi.mock('../src/hooks/useAPI.tsx', () => ({
    useGetClusterDescription: vi.fn(),
    useGetDeviceOperationListPerf: vi.fn(),
    useOpToPerfIdFiltered: vi.fn(),
    useOperationListRange: vi.fn(),
    useOperationsList: vi.fn(),
    usePerformanceRange: vi.fn(),
    usePerformanceReport: vi.fn(),
}));

const perfRow = (id: string, rawOpCode: string) => ({ id, raw_op_code: rawOpCode }) as PerfTableRow;

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
    (useOperationsList as Mock).mockReturnValue({ data: undefined });
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
