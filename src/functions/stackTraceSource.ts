// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { Operation } from '../model/APIData';

const FILE_PATH_REGEX = /File "(.*)"/m;
const LINE_NUMBER_REGEX = /line (\d*),/m;
// The backend reproduces this match for `/api/operations/summary`
// (`backend/ttnn_visualizer/operation_file_identifier.py`). Both run the cases in
// `tests/data/operationFileIdentifierCases.json`, so change the two together.
const OPERATION_FILE_IDENTIFIER_REGEX = /File\s+"(?:.+\/)?([^/]+)",\s+line\s+(\d+)/;

type OperationStackTraceFields = Pick<Operation, 'stack_trace' | 'stack_trace_source_file_id'>;

export interface OperationSourceData {
    filePath: string;
    lineNumber: number | null;
}

export const getStackTraceFilePath = (stackTrace: string): string => FILE_PATH_REGEX.exec(stackTrace)?.[1] ?? '';

export const getStackTraceLineNumber = (stackTrace: string): number | null => {
    const match = LINE_NUMBER_REGEX.exec(stackTrace);
    return match?.[1] ? parseInt(match[1], 10) : null;
};

/** `file:line` of a stack trace's first frame, or `''` when it has none. */
export const parseFileOperationIdentifier = (stackTrace: string): string => {
    if (!stackTrace) {
        return '';
    }

    const match = stackTrace.match(OPERATION_FILE_IDENTIFIER_REGEX);

    if (match) {
        return `${match[1]}:${match[2]}`;
    }

    return '';
};

export const extractOperationSourceData = (operation: OperationStackTraceFields): OperationSourceData | null => {
    const filePath = getStackTraceFilePath(operation.stack_trace);
    const hasSource = !!filePath || operation.stack_trace_source_file_id != null;

    if (!hasSource) {
        return null;
    }

    return {
        filePath,
        lineNumber: getStackTraceLineNumber(operation.stack_trace),
    };
};
