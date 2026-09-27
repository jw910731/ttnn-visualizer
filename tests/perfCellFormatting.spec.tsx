// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import '@testing-library/jest-dom/vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, describe, expect, it } from 'vitest';
import { BoundType, ColumnDefinition, ColumnKeys } from '../src/definitions/PerfTable';
import { OpType } from '../src/definitions/Performance';
import ROUTES from '../src/definitions/Routes';
import {
    formatCell,
    formatCellText,
    getOperationDetails,
    getOperationsById,
    isSuppressedCell,
} from '../src/functions/perfFunctions';
import { OperationDescription } from '../src/model/APIData';
import { TypedPerfTableRow } from '../src/model/PerfTable';

const OP_COLUMN: ColumnDefinition = { name: 'OP', key: ColumnKeys.OP };
const DEVICE_TIME_COLUMN: ColumnDefinition = {
    name: 'Device Time',
    key: ColumnKeys.DeviceTime,
    unit: 'µs',
    decimals: 0,
};

const operation = (id: number, name: string, operationFileIdentifier: string): OperationDescription =>
    ({ id, name, operationFileIdentifier }) as unknown as OperationDescription;

const makeRow = (overrides: Partial<TypedPerfTableRow>): TypedPerfTableRow =>
    ({
        id: 1,
        op_type: OpType.DEVICE_OP,
        op_code: 'Matmul',
        raw_op_code: 'Matmul',
        advice: [],
        bound: null,
        isFirstHashOccurrence: true,
        ...overrides,
    }) as unknown as TypedPerfTableRow;

afterEach(cleanup);

describe('operation lookup for the OP column', () => {
    it('describes an operation by id, keeping the first of any repeated id as the linear search did', () => {
        const operationsById = getOperationsById([
            operation(11, 'matmul_op', 'matmul.cpp'),
            operation(12, 'add_op', 'add.cpp'),
            operation(11, 'shadowed_op', 'shadowed.cpp'),
        ]);

        expect(getOperationDetails(operationsById, 11)).toBe('matmul_op (matmul.cpp)');
        expect(getOperationDetails(operationsById, 12)).toBe('add_op (add.cpp)');
        expect(getOperationDetails(operationsById, 99)).toBe('');
    });

    it('links the OP cell to its operation once operations are loaded', () => {
        const operationsById = getOperationsById([operation(1234, 'matmul_op', 'matmul.cpp')]);

        render(<MemoryRouter>{formatCell(makeRow({ op: 1234 }), OP_COLUMN, operationsById)}</MemoryRouter>);

        expect(screen.getByRole('link', { name: '1234' })).toHaveAttribute('href', `${ROUTES.OPERATIONS}/1234`);
    });

    it('prints the OP cell as a plain number while operations are still loading', () => {
        render(<MemoryRouter>{formatCell(makeRow({ op: 1234 }), OP_COLUMN, null)}</MemoryRouter>);

        expect(screen.queryByRole('link')).not.toBeInTheDocument();
        expect(screen.getByText('1,234')).toBeInTheDocument();
    });
});

describe('shared cell formatting', () => {
    it('formats text cells the same way the table renders them', () => {
        render(<>{formatCell(makeRow({ device_time: 12345.6 }), DEVICE_TIME_COLUMN)}</>);

        expect(screen.getByText(formatCellText(12345.6, DEVICE_TIME_COLUMN))).toHaveTextContent('12,346 µs');
    });

    it('suppresses cells a row type does not populate', () => {
        expect(isSuppressedCell(makeRow({ device_time: 5, bound: BoundType.HOST }), ColumnKeys.DeviceTime)).toBe(true);
        expect(isSuppressedCell(makeRow({ device_time: 5, op_type: OpType.SIGNPOST }), ColumnKeys.DeviceTime)).toBe(
            true,
        );
        expect(isSuppressedCell(makeRow({ l1_fullness_percent: 50 }), ColumnKeys.L1Fullness, false)).toBe(true);
        expect(isSuppressedCell(makeRow({ device_time: null }), ColumnKeys.DeviceTime)).toBe(true);
        expect(isSuppressedCell(makeRow({ device_time: 5 }), ColumnKeys.DeviceTime)).toBe(false);
    });
});
