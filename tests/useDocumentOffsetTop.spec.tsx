// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import '@testing-library/jest-dom/vitest';
import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import useDocumentOffsetTop from '../src/hooks/useDocumentOffsetTop';

const INITIAL_TOP_PX = 250;
const SCROLL_Y_PX = 1000;

let elementTop = INITIAL_TOP_PX;
let resizeCallback: (() => void) | null = null;
const observe = vi.fn();
const disconnect = vi.fn();

const ResizeObserverStub = vi.fn(function createResizeObserver(callback: () => void) {
    resizeCallback = callback;

    return { observe, disconnect };
});

const getObservedElements = () => observe.mock.calls.map(([element]) => element as Element);

function OffsetProbe() {
    const [ref, offsetTop] = useDocumentOffsetTop<HTMLDivElement>();

    return (
        <section>
            <div
                ref={ref}
                data-testid='anchor'
            >
                {offsetTop}
            </div>
        </section>
    );
}

beforeEach(() => {
    elementTop = INITIAL_TOP_PX;
    resizeCallback = null;
    observe.mockClear();
    disconnect.mockClear();
    vi.stubGlobal('ResizeObserver', ResizeObserverStub);
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(() => ({ top: elementTop }) as DOMRect);
    Object.defineProperty(window, 'scrollY', { configurable: true, writable: true, value: SCROLL_Y_PX });
});

afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    Object.defineProperty(window, 'scrollY', { configurable: true, writable: true, value: 0 });
});

describe('useDocumentOffsetTop', () => {
    it('measures the offset from the top of the document, not the viewport, before paint', () => {
        render(<OffsetProbe />);

        expect(screen.getByTestId('anchor')).toHaveTextContent(String(INITIAL_TOP_PX + SCROLL_Y_PX));
    });

    it('watches every ancestor, since content above the element resizes one of them', () => {
        render(<OffsetProbe />);

        const anchor = screen.getByTestId('anchor');

        expect(getObservedElements()).toContain(anchor.parentElement);
        expect(getObservedElements()).toContain(document.body);
        expect(getObservedElements()).not.toContain(anchor);
    });

    it('stops observing on unmount', () => {
        const { unmount } = render(<OffsetProbe />);

        unmount();

        expect(disconnect).toHaveBeenCalled();
    });

    it('re-measures when an ancestor resizes, ignoring sub-pixel drift', () => {
        render(<OffsetProbe />);

        elementTop = INITIAL_TOP_PX + 0.2;
        act(() => resizeCallback?.());

        expect(screen.getByTestId('anchor')).toHaveTextContent(String(INITIAL_TOP_PX + SCROLL_Y_PX));

        elementTop = INITIAL_TOP_PX + 80;
        act(() => resizeCallback?.());

        expect(screen.getByTestId('anchor')).toHaveTextContent(String(INITIAL_TOP_PX + 80 + SCROLL_Y_PX));
    });
});
