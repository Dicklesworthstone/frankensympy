"""Eager implementation records for the scalar owners used by Poly.eval.

Import this with polytools, before the evaluator is loaded on its first call.
Exact class identity alone cannot make a patched Symbol.__hash__ or
Basic.__eq__ safe to use while admitting the retained source. Records include
implementation bodies and inherited owners, and comparisons invoke no public
symbolic protocol. These are local admission records, not portable proofs or
a sandbox against arbitrary replacement of Python module globals.
"""

from ..core import (
    Add, Basic, Dummy, Expr, Float, Half, Integer, Mul, NegativeOne, One,
    Pow, Rational, Symbol, Zero,
)
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record


_MRO = type.__dict__["__mro__"]
_DICT = type.__dict__["__dict__"]
_ABSENT = object()
_EXTRA_NAMES = ("free_symbols", "subs", "__ne__", "_as_python_float",
                "_assumptions", "_dummy_name", "_dummy_number")


def _extra_record(owner):
    namespace = _DICT.__get__(owner)
    return tuple(_implementation(namespace.get(name, _ABSENT)) for name in _EXTRA_NAMES)


def _records():
    classes = (Basic, Expr, Add, Mul, Pow, Integer, Rational, Symbol, Dummy,
               Float, Zero, One, NegativeOne, Half)
    owners = []
    for cls in classes:
        for owner in _MRO.__get__(cls):
            if not any(owner is previous for previous in owners):
                owners.append(owner)
    return tuple((owner, _MRO.__get__(owner), _owner_record(owner), _extra_record(owner))
                 for owner in owners)


_RECORDS = _records()


def require_original_owners():
    """Refuse altered builtin implementations before invoking their hooks."""
    for owner, recorded_mro, definitions, extra in _RECORDS:
        current_mro = _MRO.__get__(owner)
        if (len(current_mro) != len(recorded_mro)
                or any(left is not right for left, right in zip(current_mro, recorded_mro))
                or not _owner_matches(owner, definitions)):
            raise TypeError("polynomial evaluation requires original built-in scalar owners")
        try:
            current = _extra_record(owner)
        except ValueError as error:
            raise TypeError("polynomial evaluation requires original built-in scalar owners") from error
        if any(len(tokens) != len(expected)
               or any(left is not right for left, right in zip(tokens, expected))
               for tokens, expected in zip(current, extra)):
            raise TypeError("polynomial evaluation requires original built-in scalar owners")
