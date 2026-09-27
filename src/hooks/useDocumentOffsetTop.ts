// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { useLayoutEffect, useState } from 'react';

// Fractional layout jitters by less than a pixel; re-rendering a virtualised list for it buys nothing.
const OFFSET_TOLERANCE_PX = 0.5;

/**
 * Tracks an element's distance from the top of the document, for a window virtualiser's
 * `scrollMargin`. Anything above the element that changes height (a toolbar wrapping, a notice
 * appearing, a filter row growing) resizes one of its ancestors, so those are what is observed.
 * Returns a callback ref for the element and its current offset.
 */
function useDocumentOffsetTop<T extends HTMLElement>(): [(element: T | null) => void, number] {
    const [element, setElement] = useState<T | null>(null);
    const [offsetTop, setOffsetTop] = useState(0);

    useLayoutEffect(() => {
        if (!element) {
            return undefined;
        }

        const updateOffsetTop = () => {
            const nextOffsetTop = element.getBoundingClientRect().top + window.scrollY;

            setOffsetTop((currentOffsetTop) =>
                Math.abs(currentOffsetTop - nextOffsetTop) < OFFSET_TOLERANCE_PX ? currentOffsetTop : nextOffsetTop,
            );
        };

        // Measured before paint so the first frame does not lay rows out against a zero margin
        updateOffsetTop();

        if (typeof window.ResizeObserver !== 'function') {
            return undefined;
        }

        // Feature-detected immediately above, and the offset was already measured once, so a browser
        // without it keeps that offset until the next remount.
        // eslint-disable-next-line compat/compat
        const observer = new window.ResizeObserver(updateOffsetTop);

        for (let ancestor = element.parentElement; ancestor !== null; ancestor = ancestor.parentElement) {
            observer.observe(ancestor);
        }

        return () => observer.disconnect();
    }, [element]);

    return [setElement, offsetTop];
}

export default useDocumentOffsetTop;
