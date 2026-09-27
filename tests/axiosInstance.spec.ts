// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AxiosAdapter, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import axiosInstance from '../src/libs/axiosInstance';
import Endpoints from '../src/definitions/Endpoints';

const NOT_JSON = '<html>not json</html>';
const OPERATIONS = [{ id: 1, name: 'ttnn.add' }];
const OPERATIONS_ENDPOINTS = [Endpoints.OPERATIONS_LIST, Endpoints.OPERATIONS_SUMMARY];

// Answers each request with the next body as raw text, so axios parses it as it would the
// server's: a body that is not JSON reaches the interceptor as a string.
const getAdapterAnswering = (...bodies: string[]) =>
    vi.fn<AxiosAdapter>((config: InternalAxiosRequestConfig) =>
        Promise.resolve({
            data: bodies.shift(),
            status: 200,
            statusText: 'OK',
            headers: {},
            config,
        } as AxiosResponse),
    );

beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
});

describe('axiosInstance operations retry', () => {
    it.each(OPERATIONS_ENDPOINTS)('asks %s again when its body arrives as a string', async (endpoint) => {
        const adapter = getAdapterAnswering(NOT_JSON, JSON.stringify(OPERATIONS));

        const request = axiosInstance.get(endpoint, { adapter });
        await vi.runAllTimersAsync();

        await expect(request).resolves.toMatchObject({ data: OPERATIONS });
        expect(adapter).toHaveBeenCalledTimes(2);
    });

    it.each(OPERATIONS_ENDPOINTS)('fails %s once its retries run out', async (endpoint) => {
        const adapter = getAdapterAnswering(NOT_JSON, NOT_JSON, NOT_JSON, NOT_JSON);

        const request = axiosInstance.get(endpoint, { adapter });
        const failure = expect(request).rejects.toThrow(/Invalid response format/);
        await vi.runAllTimersAsync();

        await failure;
        expect(adapter).toHaveBeenCalledTimes(4);
    });

    it('leaves the body of any other endpoint alone', async () => {
        const adapter = getAdapterAnswering(NOT_JSON);

        await expect(axiosInstance.get(Endpoints.DEVICES, { adapter })).resolves.toMatchObject({ data: NOT_JSON });
        expect(adapter).toHaveBeenCalledTimes(1);
    });
});
