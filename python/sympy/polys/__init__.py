"""Polynomial module for FrankenSymPy (WS08, WS09)."""

from .polytools import (
    LC,
    Poly,
    degree,
    discriminant,
    factor,
    factor_list,
    gcd,
    groebner,
    lcm,
    monic,
    resultant,
    roots,
    sqf_list,
    sqf_part,
)

__all__ = [
    "LC",
    "Poly",
    "degree",
    "discriminant",
    "factor",
    "factor_list",
    "gcd",
    "groebner",
    "lcm",
    "monic",
    "resultant",
    "roots",
    "sqf_list",
    "sqf_part",
]
