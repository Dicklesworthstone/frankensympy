"""Latin and Greek letters as Symbols (upstream ``sympy.abc`` surface)."""

from __future__ import annotations

from .core import symbols

a, b, c, d, e, f, g, h, i, j = symbols('a, b, c, d, e, f, g, h, i, j')
k, l, m, n, o, p, q, r, s, t = symbols('k, l, m, n, o, p, q, r, s, t')
u, v, w, x, y, z = symbols('u, v, w, x, y, z')

A, B, C, D, E, F, G, H, I, J = symbols('A, B, C, D, E, F, G, H, I, J')
K, L, M, N, O, P, Q, R, S, T = symbols('K, L, M, N, O, P, Q, R, S, T')
U, V, W, X, Y, Z = symbols('U, V, W, X, Y, Z')

alpha, beta, gamma, delta = symbols('alpha, beta, gamma, delta')
epsilon, zeta, eta, theta = symbols('epsilon, zeta, eta, theta')
iota, kappa, lamda, mu = symbols('iota, kappa, lamda, mu')
nu, xi, omicron, pi = symbols('nu, xi, omicron, pi')
rho, sigma, tau, upsilon = symbols('rho, sigma, tau, upsilon')
phi, chi, psi, omega = symbols('phi, chi, psi, omega')

_clash1: dict = {name: symbols(name) for name in ("C", "E", "I", "N", "O", "Q", "S")}
_clash2: dict = {name: symbols(name) for name in (
    "beta", "zeta", "gamma", "pi", "Lambda", "lambda")}
_clash: dict = {**_clash1, **_clash2}
