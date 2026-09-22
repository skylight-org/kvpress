"""Denominator-only Bernoulli bound: the ONLY implementation of the scheme and its certificate.

Maths: phase2/DERIVATION_denominator_bernoulli.pdf. Section numbers below refer to it.
float64 torch, one KV head at a time.

Deliberately separate from e9/cert.py, which implements the full-attention ladder B0-B4. This is a
different and smaller claim: |B_hat - B| / B only. No numerator, no chi, no ratio inversion.

Shapes:  tau [n]   omega [Q, n]
"""

import math

import torch


# ---------------------------------------------------------------- §2 the score
def score_tau(omega_cal):
    """tau_i = max over calibration queries of omega_i.  omega_cal [Q, n] -> tau [n].

    The caller is responsible for having pooled the GQA query heads into omega_cal: the kept set is
    per KV head, so tau must dominate every query head that shares it (§2 applies per head, and a
    shared cache must satisfy all of them at once)."""
    return omega_cal.max(dim=0).values


# ---------------------------------------------------------------- §3 the sampling scheme
def keep_prob(tau, c):
    """r_i = min(1, c tau_i)."""
    return (c * tau).clamp(max=1.0)


def expected_memory(tau, c):
    """M(c) = sum_i min(1, c tau_i); continuous and non-decreasing in c (§3)."""
    return keep_prob(tau, c).sum().item()


def solve_c_for_memory(tau, M, iters=100):
    """Smallest c with M(c) >= M. Bisection in log c, since c spans orders of magnitude.

    At c = 1/min(tau) every token is kept, so M(c) = n: that is the upper bracket."""
    pos = tau[tau > 0]
    n_pos = pos.numel()
    M = min(float(M), float(n_pos))
    assert M > 0, "budget must be positive"
    lo, hi = 1e-12 / pos.max().item(), 1.0 / pos.min().item()
    if expected_memory(tau, lo) >= M:
        return lo
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if expected_memory(tau, mid) >= M:
            hi = mid
        else:
            lo = mid
    return hi


def sample_keep(r, generator, trials=1):
    """Z ~ Bernoulli(r), independent across i. -> bool [trials, n]."""
    return torch.rand((trials, r.numel()), generator=generator, dtype=r.dtype, device=r.device) < r


def kernel_bias(r):
    """b_i = log(1/r_i): adding this to the logit and taking a softmax denominator over the kept set
    reproduces B_hat exactly (§4, Horvitz-Thompson with weights 1/r_i)."""
    return -torch.log(r)


# ---------------------------------------------------------------- §6-§7 range and variance
def tail_constants(tau, c):
    """(a, b, nu_L, n_tail) for the tail L = {i : c tau_i < 1} (§6, §7).

        a    = max over L of max((1 - c tau_i)/c, tau_i)      < 1/c        the range
        nu_L = min(1, sum_L (1 - c tau_i) tau_i)                            the variance mass
        b    = nu_L / c                                       >= sum E Y_i^2

    Heavy hitters (c tau_i >= 1) are excluded: they are kept outright and contribute exactly zero.
    If the tail is empty, a = b = 0 and the estimator is exact."""
    r = keep_prob(tau, c)
    L = (r < 1.0) & (tau > 0)
    if not bool(L.any()):
        return 0.0, 0.0, 0.0, 0
    tL = tau[L]
    a = torch.maximum((1.0 - c * tL) / c, tL).max().item()
    nu = min(1.0, ((1.0 - c * tL) * tL).sum().item())
    return a, nu / c, nu, int(L.sum())


# ---------------------------------------------------------------- §8-§10 the certificate
def eps_bound(tau, c, delta):
    """§10: the certified relative denominator error.

        |B_hat - B| / B  <=  u + sqrt(u^2 + gamma u),   u = a l / 3,  gamma = 6 b / a,  l = log(2/delta)

    Written via (a, b) rather than the crude (1/c, nu/c) so the measured tail range is used; with
    a = 1/c and b = nu/c this reduces to l/(3c) + sqrt(l^2/(9c^2) + 2 nu l / c), the form in §10.
    Returns 0.0 when the tail is empty (the estimator is then exact)."""
    a, b, _, n_tail = tail_constants(tau, c)
    if n_tail == 0:
        return 0.0
    ell = math.log(2.0 / delta)
    u = a * ell / 3.0
    return u + math.sqrt(u * u + 2.0 * b * ell)


def solve_c_for_eps(tau, eps, delta, iters=100):
    """§11: the smallest c whose certificate is <= eps, by bisection.

    eps_bound is non-increasing in c -- as c grows the tail only shrinks and every surviving term
    carries a smaller (1 - c tau_i), while a < 1/c falls -- so bisection is exact. This is the
    monotonicity the PDF relies on to say 'no grid search needed'."""
    pos = tau[tau > 0]
    lo, hi = 1e-12 / pos.max().item(), 1.0 / pos.min().item()
    if eps_bound(tau, lo, delta) <= eps:
        return lo
    assert eps_bound(tau, hi, delta) <= eps, "no tail at c = 1/min(tau), so eps must be 0 there"
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if eps_bound(tau, mid, delta) <= eps:
            hi = mid
        else:
            lo = mid
    return hi


def certified_memory(tau, eps, delta):
    """Memory needed to certify eps: c* from §11, then M(c*)."""
    c = solve_c_for_eps(tau, eps, delta)
    return c, expected_memory(tau, c)


def memory_highprob(r, delta_N):
    """§11 realised memory: N <= M + sqrt(2 M log(1/d)) + (2/3) log(1/d) w.p. >= 1 - delta_N.

    Uses var(N) = sum r(1-r) <= M with range 1 (the PDF's stated form)."""
    ell = math.log(1.0 / delta_N)
    M = r.sum().item()
    return M + math.sqrt(2.0 * M * ell) + 2.0 * ell / 3.0


# ---------------------------------------------------------------- estimator (for the Monte Carlo)
def b_hat(Z, r, w):
    """Horvitz-Thompson denominator. Z bool [trials, n], r [n], w [Q, n] -> B_hat [trials, Q].

    Computed as (Z/r) @ w^T, which is exactly sum_i (Z_i/r_i) w_i(q)."""
    return (Z.to(w.dtype) / r) @ w.T


def rel_error(Z, r, w):
    """|B_hat - B| / B, elementwise over (trial, query). -> [trials, Q]."""
    B = w.sum(dim=1)
    return ((b_hat(Z, r, w) - B).abs() / B)
