"""Logic and Boolean algebra submodule for FrankenSymPy."""

from __future__ import annotations

from .boolalg import (
    And,
    Boolean,
    BooleanAtom,
    BooleanFalse,
    BooleanFunction,
    BooleanTrue,
    Equivalent,
    Implies,
    Not,
    Or,
    Xor,
    false,
    simplify_logic,
    to_cnf,
    to_dnf,
    true,
)
from .inference import satisfiable

__all__ = [
    "And",
    "Boolean",
    "BooleanAtom",
    "BooleanFalse",
    "BooleanFunction",
    "BooleanTrue",
    "Equivalent",
    "Implies",
    "Not",
    "Or",
    "Xor",
    "false",
    "satisfiable",
    "simplify_logic",
    "to_cnf",
    "to_dnf",
    "true",
]
