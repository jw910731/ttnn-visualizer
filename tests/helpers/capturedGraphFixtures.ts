// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { type Node, NodeType } from '../../src/model/APIData';

// Tensor and `function_end` nodes take ids from ranges of their own, clear of the
// small frame ids fixtures use, so no node can shadow another under one id —
// `processInputsOutputs` keeps only the last node written under each.
const TENSOR_NODE_ID_BASE = 1000;
const FUNCTION_END_NODE_ID_BASE = 2000;

const TENSOR_SHAPE = 'Shape([1, 32])';

/** A tensor node whose `tensor_id` is `tensorId`. */
export const capturedTensor = (tensorId: number): Node =>
    ({
        id: TENSOR_NODE_ID_BASE + tensorId,
        node_type: NodeType.tensor,
        params: { tensor_id: tensorId, shape: TENSOR_SHAPE },
        connections: [],
        stacking_level: 0,
    }) as unknown as Node;

/** A tensor node the capture recorded with a `tensor_id` that is not a number. */
export const UNREADABLE_CAPTURED_TENSOR = {
    id: TENSOR_NODE_ID_BASE - 1,
    node_type: NodeType.tensor,
    params: { tensor_id: 'nope', shape: TENSOR_SHAPE },
    connections: [],
    stacking_level: 0,
} as unknown as Node;

/** A tensor id, or a tensor node to use as it is. */
export type CapturedTensorSpec = number | Node;

export interface CapturedFrameSpec {
    id: number;
    name: string;
    consumes?: CapturedTensorSpec[];
    produces?: CapturedTensorSpec[];
}

const tensorNodeOf = (spec: CapturedTensorSpec): Node => (typeof spec === 'number' ? capturedTensor(spec) : spec);

/**
 * One device-operation frame as the capture records it, before `processInputsOutputs`
 * resolves it: the `function_start` lists the tensor nodes it consumes, and connects
 * to its `function_end`, which connects to the tensor nodes it produced. Concatenate
 * several into an operation's `device_operations`.
 */
export const capturedFrame = ({ id, name, consumes = [], produces = [] }: CapturedFrameSpec): Node[] => {
    const inputs = consumes.map(tensorNodeOf);
    const outputs = produces.map(tensorNodeOf);
    const functionEndId = FUNCTION_END_NODE_ID_BASE + id;

    return [
        ...inputs,
        {
            id,
            node_type: NodeType.function_start,
            params: { name },
            input_tensors: inputs.map((tensor) => tensor.id),
            connections: [functionEndId],
            arguments: [],
            stack_trace: [],
            stacking_level: 0,
        },
        {
            id: functionEndId,
            node_type: NodeType.function_end,
            params: { name },
            connections: outputs.map((tensor) => tensor.id),
            stacking_level: 0,
        },
        ...outputs,
    ] as unknown as Node[];
};
