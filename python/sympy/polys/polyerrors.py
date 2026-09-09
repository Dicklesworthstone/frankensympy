"""Public polynomial exceptions used by the compatibility shell."""


class BasePolynomialError(Exception):
    """Base class for polynomial-related errors."""

    def new(self, *args):
        raise NotImplementedError("abstract base class")


class GeneratorsError(BasePolynomialError):
    """Invalid polynomial generators."""
