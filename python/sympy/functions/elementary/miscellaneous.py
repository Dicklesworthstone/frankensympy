"""Max and Min (upstream sympy.functions.elementary.miscellaneous)."""

from __future__ import annotations

from typing import Any

from ...core import Function, _NativeFunction


class _MinMaxBase(_NativeFunction):
    __slots__ = ()
    _pick_greater = True

    @classmethod
    def eval(cls, *args: Any) -> Any:
        from ...core import _real_sign_of_difference, sympify

        if not args:
            raise ValueError("The Max/Min functions must have arguments.")
        flat: list = []
        for a in (sympify(a) for a in args):
            flat.extend(a.args if type(a) is cls else [a])
        numeric = [a for a in flat if not a.free_symbols]
        symbolic = []
        for a in flat:
            if a.free_symbols and a not in symbolic:
                symbolic.append(a)
        best = None
        undecided = []
        for a in numeric:
            if best is None:
                best = a
                continue
            try:
                sign = _real_sign_of_difference(a, best)
            except TypeError:
                raise ValueError(f"The argument '{a}' is not comparable.")
            if sign is None:
                if a != best and a not in undecided:
                    undecided.append(a)
                continue
            if (sign > 0) == cls._pick_greater and sign != 0:
                best = a
        kept = ([best] if best is not None else []) + undecided + symbolic
        if len(kept) == 1:
            return kept[0]
        from ...printing.str import sort_key

        kept.sort(key=sort_key)
        if len(kept) == len(args) and all(k is a for k, a in zip(kept, args)):
            return None
        return Function.__new__(cls, *kept, evaluate=False)


class Max(_MinMaxBase):
    """Maximum of the arguments; numeric arguments reduce to one."""

    __slots__ = ()
    _pick_greater = True


class Min(_MinMaxBase):
    """Minimum of the arguments; numeric arguments reduce to one."""

    __slots__ = ()
    _pick_greater = False


__all__ = ["Max", "Min"]
