# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
The ``file:line`` label the client gives an operation, computed server-side.

``parseFileOperationIdentifier`` (``src/functions/stackTraceSource.ts``) takes the
first match of::

    File\\s+"(?:.+\\/)?([^/]+)",\\s+line\\s+(\\d+)

in the stack trace. The summary route needs the same label without shipping every
trace, so this reproduces that match. It does not run the regex through ``re``:
backtracking makes it quadratic in the length of a line holding many ``File "``
that each fail -- 3 s for one crafted 70 KB trace, minutes for a report of them --
and this runs on uploaded reports, under ``SERVER_MODE`` too, on every page load.
Instead the regex's own choices are made directly, scanning each stretch of the
trace a bounded number of times:

* A frame starts at ``File``, whitespace and ``"``; starts are tried in order.
* A frame ends at ``",``, whitespace, ``line``, whitespace and a digit. None of that
  is ``/``, so a frame end lies inside one run of non-``/`` characters, and whether
  a position ends a frame depends on that position alone.
* ``(?:.+/)?`` is tried first. ``.+`` is greedy and stops at a line terminator, so
  it ends at the last ``/`` on the start's line whose run holds a frame end at
  least one character on (``([^/]+)`` cannot be empty); that answer is the same
  for every start on the line that precedes the ``/``. ``([^/]+)`` is greedy too,
  so the frame end is the last one in the run.
* Failing that, ``([^/]+)`` starts right after the quote, with the same rule.

JavaScript's ``\\s``, ``.`` and ``\\d`` are not Python's, so the classes are spelled
out: JavaScript's ``.`` stops at ``\\r``, U+2028 and U+2029 as well as ``\\n``; its
``\\s`` takes U+FEFF but not U+001C-U+001F or U+0085; its ``\\d`` is ASCII only.
``tests/data/operationFileIdentifierCases.json`` holds cases whose expectations the
JavaScript function produced; both sides run them.
"""

import re
from typing import Dict, Optional, Tuple

JS_WHITESPACE = (
    r"[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]"
)
JS_LINE_TERMINATOR = r"[\n\r\u2028\u2029]"

_FRAME_START = re.compile(rf'File{JS_WHITESPACE}+"')
_FRAME_END = re.compile(rf'",{JS_WHITESPACE}+line{JS_WHITESPACE}+([0-9]+)')
_LINE_TERMINATOR = re.compile(JS_LINE_TERMINATOR)
_NOT_FOUND = -1


def operation_file_identifier(stack_trace: Optional[str]) -> str:
    """``file:line`` of the first frame in a stack trace, ``""`` when there is none."""
    if not stack_trace:
        return ""
    return _FirstFrameSearch(stack_trace).identifier()


class _FirstFrameSearch:
    def __init__(self, text: str):
        self._text = text
        self._last_frame_end_by_run_end: Dict[int, int] = {}

    def identifier(self) -> str:
        text = self._text
        line_end = _NOT_FOUND
        line_match: Optional[Tuple[int, int]] = None
        next_slash = _NOT_FOUND

        for frame_start in _FRAME_START.finditer(text):
            name_start = frame_start.end()

            if name_start > line_end:
                terminator = _LINE_TERMINATOR.search(text, name_start)
                line_end = terminator.start() if terminator else len(text)
                line_match = self._last_usable_slash(name_start, line_end)

            # `(?:.+/)?` taken. `.+` needs a character, so the `/` is past the quote.
            if line_match is not None and line_match[0] > name_start:
                slash, frame_end = line_match
                return self._identifier(slash + 1, frame_end)

            # `(?:.+/)?` skipped: `([^/]+)` starts at the quote's end.
            if name_start < len(text) and text[name_start] != "/":
                if next_slash < name_start:
                    next_slash = self._next_slash(name_start)
                frame_end = self._last_frame_end_in_run(next_slash)
                if frame_end > name_start:
                    return self._identifier(name_start, frame_end)

        return ""

    def _last_usable_slash(
        self, name_start: int, line_end: int
    ) -> Optional[Tuple[int, int]]:
        """The last `/` on the line past ``name_start`` that can end `.+`, with its frame end."""
        slash = self._text.rfind("/", name_start + 1, line_end)
        run_end = self._next_slash(slash + 1) if slash != _NOT_FOUND else _NOT_FOUND
        while slash != _NOT_FOUND:
            frame_end = self._last_frame_end_in_run(run_end)
            if frame_end >= slash + 2:
                return slash, frame_end
            run_end = slash
            slash = self._text.rfind("/", name_start + 1, slash)
        return None

    def _next_slash(self, position: int) -> int:
        slash = self._text.find("/", position)
        return len(self._text) if slash == _NOT_FOUND else slash

    def _last_frame_end_in_run(self, run_end: int) -> int:
        """
        The last frame end in the run of non-`/` characters that ends at
        ``run_end``. Memoised by run, since starts sharing a run ask again.
        """
        cached = self._last_frame_end_by_run_end.get(run_end)
        if cached is not None:
            return cached
        run_start = self._text.rfind("/", 0, run_end) + 1
        last_frame_end = _NOT_FOUND
        for frame_end in _FRAME_END.finditer(self._text, run_start, run_end):
            last_frame_end = frame_end.start()
        self._last_frame_end_by_run_end[run_end] = last_frame_end
        return last_frame_end

    def _identifier(self, name_start: int, frame_end: int) -> str:
        match = _FRAME_END.match(self._text, frame_end)
        assert match is not None, "a frame end is only ever where `_FRAME_END` matched"
        return f"{self._text[name_start:frame_end]}:{match.group(1)}"
