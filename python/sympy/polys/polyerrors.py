"""Public polynomial exceptions used by the compatibility shell."""


class BasePolynomialError(Exception):
    """Base class for polynomial-related errors."""

    def new(self, *args):
        raise NotImplementedError("abstract base class")


class GeneratorsError(BasePolynomialError):
    """Invalid polynomial generators."""


class PolynomialError(BasePolynomialError):
    """An expression is not a valid polynomial in the given generators."""


class NotInvertible(PolynomialError):
    """An element has no multiplicative inverse in the specified quotient."""


class CoercionFailed(BasePolynomialError):
    """A coefficient cannot be represented in the requested exact domain."""


class UnificationFailed(BasePolynomialError):
    """The operands do not belong to a common supported polynomial ring."""


class DomainError(BasePolynomialError):
    """The requested polynomial coefficient domain is invalid."""


class GeneratorsNeeded(GeneratorsError):
    """A constant polynomial requires an explicit generator."""


class ExactQuotientFailed(PolynomialError):
    """Polynomial division has a nonzero remainder."""
