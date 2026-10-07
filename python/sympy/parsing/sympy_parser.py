"""``parse_expr`` with upstream-style token transformations.

Transformations are callables ``(tokens, local_dict, global_dict) ->
tokens`` over Python ``tokenize`` tokens, as upstream; the standard set
is applied by the expression constructor itself, so
``standard_transformations`` is a marker. ``implicit_multiplication``,
``implicit_application``, ``implicit_multiplication_application`` and
``convert_xor`` rewrite the token stream before construction.
"""

from __future__ import annotations

import io
import keyword
import tokenize
from typing import Any


def _known_callable(name: str, local_dict: dict, global_dict: dict) -> bool:
    obj = local_dict.get(name, global_dict.get(name))
    if obj is None:
        import sympy

        obj = getattr(sympy, name, None)
    from ..core import Symbol

    return callable(obj) and not isinstance(obj, Symbol)


def auto_symbol(tokens: list, local_dict: dict, global_dict: dict) -> list:
    return tokens


def auto_number(tokens: list, local_dict: dict, global_dict: dict) -> list:
    return tokens


def factorial_notation(tokens: list, local_dict: dict, global_dict: dict) -> list:
    return tokens


def convert_xor(tokens: list, local_dict: dict, global_dict: dict) -> list:
    """``^`` means exponentiation."""
    return [(tokenize.OP, "**") if (t == tokenize.OP and v == "^") else (t, v) for t, v in tokens]


def implicit_multiplication(tokens: list, local_dict: dict, global_dict: dict) -> list:
    """Insert ``*`` between juxtaposed operands: ``2x``, ``x y``, ``)(``,
    ``2(x)``; a known function name followed by ``(`` stays a call."""
    out: list = []
    for i, (t, v) in enumerate(tokens):
        if out:
            pt, pv = out[-1]
            prev_operand = pt in (tokenize.NUMBER, tokenize.NAME) or (pt == tokenize.OP and pv == ")")
            cur_operand = t in (tokenize.NUMBER, tokenize.NAME) or (t == tokenize.OP and v == "(")
            if prev_operand and cur_operand and not (pt == tokenize.NAME and keyword.iskeyword(pv)):
                is_call = pt == tokenize.NAME and v == "(" and _known_callable(pv, local_dict, global_dict)
                if not is_call and not (t == tokenize.NAME and keyword.iskeyword(v)):
                    out.append((tokenize.OP, "*"))
        out.append((t, v))
    return out


def implicit_application(tokens: list, local_dict: dict, global_dict: dict) -> list:
    """``sin x`` -> ``sin(x)`` for known function names."""
    out: list = []
    i = 0
    while i < len(tokens):
        t, v = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if (t == tokenize.NAME and nxt is not None and nxt[0] in (tokenize.NAME, tokenize.NUMBER)
                and _known_callable(v, local_dict, global_dict)):
            out.extend([(t, v), (tokenize.OP, "("), nxt, (tokenize.OP, ")")])
            i += 2
            continue
        out.append((t, v))
        i += 1
    return out


def implicit_multiplication_application(tokens: list, local_dict: dict, global_dict: dict) -> list:
    return implicit_multiplication(implicit_application(tokens, local_dict, global_dict), local_dict, global_dict)


standard_transformations = (auto_symbol, auto_number, factorial_notation)
_STANDARD = set(standard_transformations)


def _tokens(s: str) -> list:
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(s).readline):
        if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT):
            continue
        out.append((tok.type, tok.string))
    return out


def _untokenize(tokens: list) -> str:
    return " ".join(v for _, v in tokens)


def stringify_expr(s: str, local_dict: dict, global_dict: dict, transformations: Any) -> str:
    tokens = _tokens(s)
    for tr in transformations:
        if tr in _STANDARD:
            continue
        tokens = tr(tokens, local_dict, global_dict)
    return _untokenize(tokens)


def parse_expr(s: str, local_dict: Any = None, transformations: Any = standard_transformations,
               global_dict: Any = None, evaluate: bool = True) -> Any:
    """Convert the string ``s`` into an expression."""
    from ..core import sympify

    local_dict = dict(local_dict or {})
    global_dict = dict(global_dict or {})
    if transformations == "all":
        transformations = standard_transformations + (convert_xor, implicit_multiplication_application)
    elif transformations == "implicit":
        transformations = standard_transformations + (implicit_multiplication_application,)
    code = stringify_expr(s, local_dict, global_dict, transformations)
    return sympify(code, locals=local_dict or None, evaluate=evaluate)


__all__ = [
    "auto_number",
    "auto_symbol",
    "convert_xor",
    "factorial_notation",
    "implicit_application",
    "implicit_multiplication",
    "implicit_multiplication_application",
    "parse_expr",
    "standard_transformations",
    "stringify_expr",
]
