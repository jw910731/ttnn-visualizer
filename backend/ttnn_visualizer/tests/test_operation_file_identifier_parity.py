# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Pins ``operation_file_identifier`` to the frontend's ``parseFileOperationIdentifier``.

The OP column labels an operation ``name (file:line)`` from whichever copy reached it: the
performance route reads the server's, from ``/api/operations/summary``, and the operations
views compute their own from the stack trace. A divergence would label the same operation
two ways with nothing reporting it. Two checks close that:

* Both sides run ``tests/data/operationFileIdentifierCases.json``, whose expectations the
  JavaScript function produced (``tests/stackTraceSource.spec.ts`` checks they still are).
* The server does not run the regex -- see the module docstring for why -- so random traces
  built from the characters the regex cares about are compared against the regex itself,
  translated to JavaScript's classes. The oracle derives those classes from ECMA-262's
  definitions here rather than importing the module's, so a wrong class cannot move the
  implementation and its oracle together; a separate test holds the module's classes to
  the same derivation.
"""

import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import pytest
from ttnn_visualizer.operation_file_identifier import (
    JS_LINE_TERMINATOR,
    JS_WHITESPACE,
    operation_file_identifier,
)

_CASES_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "data"
    / "operationFileIdentifierCases.json"
)
_CASES = json.loads(_CASES_PATH.read_text(encoding="utf-8"))

_ALL_CODE_POINTS = [chr(code_point) for code_point in range(sys.maxunicode + 1)]

# ECMA-262 LineTerminator: LF, CR, LINE SEPARATOR and PARAGRAPH SEPARATOR.
_JS_LINE_TERMINATORS = frozenset("\n\r\u2028\u2029")
# ECMA-262 WhiteSpace: TAB, VT, FF, ZWNBSP and every `Zs` character. `\s` is WhiteSpace
# plus LineTerminator.
_JS_WHITESPACE = (
    frozenset("\t\v\f\ufeff")
    | frozenset(char for char in _ALL_CODE_POINTS if unicodedata.category(char) == "Zs")
    | _JS_LINE_TERMINATORS
)


def _character_class(chars, *, negated=False):
    return "[" + ("^" if negated else "") + re.escape("".join(sorted(chars))) + "]"


_JS_S = _character_class(_JS_WHITESPACE)
# JavaScript's `.`: anything but a line terminator.
_JS_DOT = _character_class(_JS_LINE_TERMINATORS, negated=True)
_JS_REGEX = re.compile(
    rf'File{_JS_S}+"(?:{_JS_DOT}+/)?([^/]+)",{_JS_S}+line{_JS_S}+([0-9]+)'
)

# Pieces of real frames, and the characters each class boundary turns on.
_TOKENS = [
    "File",
    'File "',
    'File "/',
    '"',
    "/",
    ",",
    '",',
    " line ",
    "line",
    " ",
    "\t",
    "\n",
    "\r",
    "\u2028",
    "\u2029",
    "\u00a0",
    "\ufeff",
    "\u3000",
    "\u1680",
    "\u2000",
    "\u200a",
    "\u202f",
    "\u205f",
    "\v",
    "\f",
    "\u001c",
    "\u001f",
    "\u0085",
    "\u180e",
    "\u200b",
    "7",
    "42",
    "\u0663",
    "a",
    "model.py",
    ", in forward",
    '", line 3',
]
_RANDOM_TRACE_COUNT = 20_000
_MAX_RANDOM_TRACE_TOKENS = 24
# One line of repeated starts that each fail, which costs the regex quadratic time:
# ~3 s at 70 KB. The bound is two orders of magnitude above what a linear scan needs.
_ADVERSARIAL_TRACE = 'File "/' * 30_000 + '\n/", line 1'
_ADVERSARIAL_TIME_LIMIT_S = 2.0


def _regex_identifier(stack_trace):
    if not stack_trace:
        return ""
    match = _JS_REGEX.search(stack_trace)
    return f"{match.group(1)}:{match.group(2)}" if match else ""


@pytest.mark.parametrize(
    "module_class,expected_chars",
    [(JS_WHITESPACE, _JS_WHITESPACE), (JS_LINE_TERMINATOR, _JS_LINE_TERMINATORS)],
    ids=["whitespace", "line-terminator"],
)
def test_the_modules_classes_are_javascripts(module_class, expected_chars):
    """Every code point, so a character added to or dropped from either class fails."""
    pattern = re.compile(module_class)
    matched = {char for char in _ALL_CODE_POINTS if pattern.fullmatch(char)}

    assert matched == expected_chars, sorted(
        f"U+{ord(char):04X}" for char in matched ^ expected_chars
    )


def test_the_shared_cases_include_a_trace_with_no_frame():
    """An empty expectation is the answer for most traces a capture can miss."""
    assert any(case["stackTrace"] and not case["expected"] for case in _CASES)


@pytest.mark.parametrize(
    "stack_trace,expected",
    [(case["stackTrace"], case["expected"]) for case in _CASES],
    ids=[case["description"] for case in _CASES],
)
def test_operation_file_identifier_matches_the_frontend(stack_trace, expected):
    assert operation_file_identifier(stack_trace) == expected


@pytest.mark.parametrize(
    "stack_trace",
    [case["stackTrace"] for case in _CASES],
    ids=[case["description"] for case in _CASES],
)
def test_the_translated_regex_matches_the_frontend_too(stack_trace):
    """The oracle below is only as good as its classes; the JavaScript cases pin them."""
    expected = next(
        case["expected"] for case in _CASES if case["stackTrace"] == stack_trace
    )
    assert _regex_identifier(stack_trace) == expected


def test_operation_file_identifier_agrees_with_the_regex_on_random_traces():
    rng = random.Random(1860)
    disagreements = []
    for _ in range(_RANDOM_TRACE_COUNT):
        trace = "".join(
            rng.choice(_TOKENS) for _ in range(rng.randint(0, _MAX_RANDOM_TRACE_TOKENS))
        )
        if operation_file_identifier(trace) != _regex_identifier(trace):
            disagreements.append(trace)
    assert not disagreements, disagreements[:5]


def test_a_missing_trace_has_no_identifier():
    """``stack_traces`` can hold NULL, which the client never sees as anything but ``''``."""
    assert operation_file_identifier(None) == ""


def test_a_trace_crafted_to_backtrack_costs_linear_time():
    started = time.perf_counter()
    identifier = operation_file_identifier(_ADVERSARIAL_TRACE)
    elapsed = time.perf_counter() - started

    assert identifier == ""
    assert elapsed < _ADVERSARIAL_TIME_LIMIT_S
