from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class HopfSearchConfig:
    d1: float = 1.0
    d2: float = -5.8
    u_bar: float = 1.0
    fu: float = -0.5
    tau: float = 2.0
    domain_factor: float = 3.0
    n_list: tuple[int, ...] = (1, 3, 5, 7)
    k_list: tuple[int, ...] = (0,)
    eps_min: float = 0.02
    eps_max_margin: float = 0.02
    num_eps: int = 220
    omega_min: float = 1.0e-4
    omega_max: float = 80.0
    num_omega: int = 12000
    min_hopf_omega: float = 1.0e-2
    max_roots_per_eps: int = 12
    max_total_branches: int = 30
    branch_gap_abs: float = 1.5
    branch_gap_rel: float = 0.15
    branch_max_fill: int = 3
    s_tol: float = 0.05
    polynomial_degree: int = 3
    polynomial_window: int = 7
    polynomial_plot_points: int = 80
    duplicate_eps_tol: float = 1.0e-5


@dataclass(frozen=True)
class HopfPoint:
    epsilon: float
    omega: float
    n: int
    k: int
    mu: float
    branch_index: int
    s_value: float


@dataclass(frozen=True)
class HopfPolynomialFit:
    n: int
    k: int
    mu: float
    branch_index: int
    epsilon: float
    omega: float
    s_value: float
    seed_epsilon: float
    eps_fit: np.ndarray
    s_fit: np.ndarray
    eps_plot: np.ndarray
    s_plot: np.ndarray


@dataclass(frozen=True)
class HopfBranchData:
    n: int
    mu: float
    eps_grid: np.ndarray
    omega_mat: np.ndarray


def g_value(
    w: np.ndarray | float,
    e: float,
    mu: float,
    d1: float,
    d2: float,
    u: float,
    fu: float,
    tau: float,
) -> np.ndarray | float:
    a = e * (tau - e) * tau / (4.0 * d2 * u * mu)
    b = e * (tau - e) * tau * (d1 * mu - fu) ** 2 / (4.0 * d2 * u * mu)
    return (
        a * np.asarray(w) ** 6
        + b * np.asarray(w) ** 4
        + e * np.sin(np.asarray(w) * tau) * np.asarray(w) ** 3
        - (d1 * mu - fu) * (tau - e + e * np.cos(np.asarray(w) * tau)) * np.asarray(w) ** 2
        + (2.0 * d2 * u * mu / tau) * (np.cos(np.asarray(w) * tau) - 1.0)
    )


def sin_rhs(w: float, e: float, mu: float, d2: float, u: float, tau: float) -> float:
    return e * (tau - e) / (2.0 * d2 * u * mu) * w**3 + (e / tau) * np.sin(w * tau)


def cos_rhs(
    w: float,
    e: float,
    mu: float,
    d1: float,
    d2: float,
    u: float,
    fu: float,
    tau: float,
) -> float:
    return (
        (tau - e) / tau
        + (e / tau) * np.cos(w * tau)
        - e * (tau - e) * (d1 * mu - fu) / (2.0 * d2 * u * mu) * w**2
    )


def s_value_for_root(
    omega: float,
    epsilon: float,
    mu: float,
    k: int,
    config: HopfSearchConfig,
) -> float:
    s = sin_rhs(omega, epsilon, mu, config.d2, config.u_bar, config.tau)
    c = cos_rhs(
        omega,
        epsilon,
        mu,
        config.d1,
        config.d2,
        config.u_bar,
        config.fu,
        config.tau,
    )
    theta = np.arctan2(s, c)
    if theta < 0:
        theta += 2.0 * np.pi
    return float(epsilon - (theta + 2.0 * np.pi * k) / omega)


def find_roots_for_fixed_eps(epsilon: float, mu: float, config: HopfSearchConfig) -> np.ndarray:
    ws = np.linspace(config.omega_min, config.omega_max, config.num_omega)
    vals = g_value(
        ws,
        epsilon,
        mu,
        config.d1,
        config.d2,
        config.u_bar,
        config.fu,
        config.tau,
    )
    roots: list[float] = []
    for i in range(ws.size - 1):
        a = float(ws[i])
        b = float(ws[i + 1])
        fa = float(vals[i])
        fb = float(vals[i + 1])
        if not np.isfinite(fa) or not np.isfinite(fb):
            continue
        if abs(fa) < 1.0e-8:
            if i == 0:
                continue
            root = a
        elif fa * fb < 0.0:
            root = brentq(
                lambda w: float(
                    g_value(w, epsilon, mu, config.d1, config.d2, config.u_bar, config.fu, config.tau)
                ),
                a,
                b,
            )
        else:
            continue

        if not roots or all(abs(root - old) > 1.0e-4 for old in roots):
            if root >= config.min_hopf_omega:
                roots.append(root)
            if len(roots) >= config.max_roots_per_eps:
                break
    return np.asarray(sorted(roots), dtype=float)


def assign_branches_track(
    roots_by_epsilon: list[np.ndarray],
    max_branches: int,
    gap_abs: float,
    gap_rel: float,
) -> np.ndarray:
    omega_mat = np.full((len(roots_by_epsilon), max_branches), np.nan, dtype=float)
    last_omega = np.full(max_branches, np.nan, dtype=float)
    n_active = 0

    for ie, roots in enumerate(roots_by_epsilon):
        if roots.size == 0:
            continue
        used = np.zeros(roots.size, dtype=bool)
        if n_active > 0:
            branch_ids = np.where(np.isfinite(last_omega[:n_active]))[0]
            if branch_ids.size > 0:
                la = last_omega[branch_ids][:, None]
                distances = np.abs(la - roots[None, :])
                thresholds = np.maximum(gap_abs, gap_rel * la)
                distances[distances > thresholds] = np.inf

                while True:
                    flat_index = int(np.argmin(distances))
                    best = float(distances.flat[flat_index])
                    if not np.isfinite(best):
                        break
                    bi, ri = np.unravel_index(flat_index, distances.shape)
                    branch_id = int(branch_ids[bi])
                    omega_mat[ie, branch_id] = roots[ri]
                    last_omega[branch_id] = roots[ri]
                    used[ri] = True
                    distances[bi, :] = np.inf
                    distances[:, ri] = np.inf

        for ri in np.where(~used)[0]:
            if n_active >= max_branches:
                break
            omega_mat[ie, n_active] = roots[ri]
            last_omega[n_active] = roots[ri]
            n_active += 1
    return omega_mat


def fill_small_gaps(omega_mat: np.ndarray, max_gap: int) -> np.ndarray:
    filled = omega_mat.copy()
    for ib in range(filled.shape[1]):
        col = filled[:, ib]
        valid = np.where(np.isfinite(col))[0]
        if valid.size < 2:
            continue
        for left, right in zip(valid[:-1], valid[1:]):
            gap = right - left - 1
            if 0 < gap <= max_gap:
                values = np.linspace(col[left], col[right], gap + 2)
                col[left + 1 : right] = values[1:-1]
        filled[:, ib] = col
    return filled


def _interpolate_zero_crossing(
    x0: float,
    y0: float,
    w0: float,
    x1: float,
    y1: float,
    w1: float,
) -> tuple[float, float]:
    if y1 == y0:
        alpha = 0.0
    else:
        alpha = -y0 / (y1 - y0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return x0 + alpha * (x1 - x0), w0 + alpha * (w1 - w0)


def _s_curve_for_branch(
    eps_grid: np.ndarray,
    omega_curve: np.ndarray,
    mu: float,
    k: int,
    config: HopfSearchConfig,
) -> np.ndarray:
    s_curve = np.full(eps_grid.shape, np.nan, dtype=float)
    for ie, omega in enumerate(omega_curve):
        if np.isfinite(omega):
            s_curve[ie] = s_value_for_root(float(omega), float(eps_grid[ie]), mu, k, config)
    return s_curve


def _local_window_indices(
    finite: np.ndarray,
    center_index: int,
    window_size: int,
    min_size: int,
) -> np.ndarray:
    if finite.size < min_size:
        return np.asarray([], dtype=int)
    window_size = max(window_size, min_size)
    window_size = min(window_size, finite.size)
    if window_size % 2 == 0 and window_size < finite.size:
        window_size += 1

    center_pos = int(np.argmin(np.abs(finite - center_index)))
    left = center_pos - window_size // 2
    right = left + window_size
    if left < 0:
        left = 0
        right = window_size
    if right > finite.size:
        right = finite.size
        left = right - window_size
    return finite[left:right]


def _refine_hopf_root_with_polynomial(
    eps_grid: np.ndarray,
    omega_curve: np.ndarray,
    s_curve: np.ndarray,
    seed_index: int,
    seed_epsilon: float,
    n: int,
    k: int,
    mu: float,
    branch_index: int,
    config: HopfSearchConfig,
    root_interval: tuple[float, float] | None = None,
) -> HopfPolynomialFit | None:
    finite = np.where(np.isfinite(s_curve) & np.isfinite(omega_curve))[0]
    degree = max(1, int(config.polynomial_degree))
    min_size = degree + 2
    fit_indices = _local_window_indices(
        finite,
        seed_index,
        int(config.polynomial_window),
        min_size,
    )
    if fit_indices.size < min_size:
        return None

    eps_fit = eps_grid[fit_indices]
    s_fit = s_curve[fit_indices]
    omega_fit = omega_curve[fit_indices]
    if np.unique(eps_fit).size < 2:
        return None

    degree = min(degree, np.unique(eps_fit).size - 1)
    x0 = float(seed_epsilon)
    s_poly = np.poly1d(np.polyfit(eps_fit - x0, s_fit, degree))
    omega_poly = np.poly1d(np.polyfit(eps_fit - x0, omega_fit, degree))

    roots = np.roots(s_poly)
    real_roots = np.asarray(
        [x0 + float(root.real) for root in roots if abs(root.imag) < 1.0e-8],
        dtype=float,
    )
    if real_roots.size == 0:
        return None

    fit_min = float(np.min(eps_fit))
    fit_max = float(np.max(eps_fit))
    real_roots = real_roots[(fit_min <= real_roots) & (real_roots <= fit_max)]
    if real_roots.size == 0:
        return None

    if root_interval is not None:
        lo, hi = root_interval
        in_interval = real_roots[(lo <= real_roots) & (real_roots <= hi)]
        if in_interval.size > 0:
            real_roots = in_interval

    root_eps = float(real_roots[np.argmin(np.abs(real_roots - seed_epsilon))])
    omega_guess = float(omega_poly(root_eps - x0))
    roots_at_eps = find_roots_for_fixed_eps(root_eps, mu, config)
    if roots_at_eps.size == 0:
        root_omega = omega_guess
    else:
        root_omega = float(roots_at_eps[np.argmin(np.abs(roots_at_eps - omega_guess))])
    s_value = s_value_for_root(root_omega, root_eps, mu, k, config)

    eps_plot = np.linspace(
        fit_min,
        fit_max,
        max(2, int(config.polynomial_plot_points)),
    )
    s_plot = s_poly(eps_plot - x0)
    return HopfPolynomialFit(
        n=n,
        k=k,
        mu=mu,
        branch_index=branch_index,
        epsilon=root_eps,
        omega=root_omega,
        s_value=s_value,
        seed_epsilon=x0,
        eps_fit=eps_fit,
        s_fit=s_fit,
        eps_plot=eps_plot,
        s_plot=s_plot,
    )


def _deduplicate_fits(
    fits: list[HopfPolynomialFit],
    config: HopfSearchConfig,
) -> list[HopfPolynomialFit]:
    deduped: list[HopfPolynomialFit] = []
    for fit in sorted(fits, key=lambda item: (item.n, item.k, item.branch_index, item.epsilon, abs(item.s_value))):
        duplicate_index = None
        for i, old in enumerate(deduped):
            if (
                old.n == fit.n
                and old.k == fit.k
                and old.branch_index == fit.branch_index
                and abs(old.epsilon - fit.epsilon) <= config.duplicate_eps_tol
            ):
                duplicate_index = i
                break
        if duplicate_index is None:
            deduped.append(fit)
        elif abs(fit.s_value) < abs(deduped[duplicate_index].s_value):
            deduped[duplicate_index] = fit
    return deduped


def compute_hopf_branches(config: HopfSearchConfig = HopfSearchConfig()) -> list[HopfBranchData]:
    eps_grid = np.linspace(config.eps_min, config.tau - config.eps_max_margin, config.num_eps)
    branches: list[HopfBranchData] = []

    for n in config.n_list:
        mu = n**2 / config.domain_factor**2
        roots_by_epsilon = [
            find_roots_for_fixed_eps(float(eps), mu, config) for eps in eps_grid
        ]
        omega_mat = assign_branches_track(
            roots_by_epsilon,
            config.max_total_branches,
            config.branch_gap_abs,
            config.branch_gap_rel,
        )
        omega_mat = fill_small_gaps(omega_mat, config.branch_max_fill)
        keep = np.any(np.isfinite(omega_mat), axis=0)
        omega_mat = omega_mat[:, keep]
        branches.append(HopfBranchData(n=n, mu=mu, eps_grid=eps_grid, omega_mat=omega_mat))

    return branches


def hopf_points_from_branches(
    branches: list[HopfBranchData],
    config: HopfSearchConfig = HopfSearchConfig(),
) -> list[HopfPoint]:
    fits = hopf_polynomial_fits_from_branches(branches, config)
    points = [
        HopfPoint(
            epsilon=fit.epsilon,
            omega=fit.omega,
            n=fit.n,
            k=fit.k,
            mu=fit.mu,
            branch_index=fit.branch_index,
            s_value=fit.s_value,
        )
        for fit in fits
    ]
    points.sort(key=lambda p: (abs(p.s_value), p.n, p.k, p.epsilon, p.omega))
    return points


def hopf_polynomial_fits_from_branches(
    branches: list[HopfBranchData],
    config: HopfSearchConfig = HopfSearchConfig(),
) -> list[HopfPolynomialFit]:
    fits: list[HopfPolynomialFit] = []
    for branch_data in branches:
        n = branch_data.n
        mu = branch_data.mu
        eps_grid = branch_data.eps_grid
        omega_mat = branch_data.omega_mat
        for k in config.k_list:
            for ib in range(omega_mat.shape[1]):
                omega_curve = omega_mat[:, ib]
                s_curve = _s_curve_for_branch(eps_grid, omega_curve, mu, k, config)

                finite = np.where(np.isfinite(s_curve))[0]
                for idx in finite:
                    if abs(s_curve[idx]) <= 1.0e-8:
                        fit = _refine_hopf_root_with_polynomial(
                            eps_grid,
                            omega_curve,
                            s_curve,
                            int(idx),
                            float(eps_grid[idx]),
                            n,
                            k,
                            mu,
                            ib,
                            config,
                        )
                        if fit is not None:
                            fits.append(fit)

                for left, right in zip(finite[:-1], finite[1:]):
                    if right != left + 1:
                        continue
                    s0 = s_curve[left]
                    s1 = s_curve[right]
                    if s0 * s1 > 0:
                        continue
                    eps, _ = _interpolate_zero_crossing(
                        float(eps_grid[left]),
                        float(s0),
                        float(omega_curve[left]),
                        float(eps_grid[right]),
                        float(s1),
                        float(omega_curve[right]),
                    )
                    fit = _refine_hopf_root_with_polynomial(
                        eps_grid,
                        omega_curve,
                        s_curve,
                        int(left if abs(s0) <= abs(s1) else right),
                        float(eps),
                        n,
                        k,
                        mu,
                        ib,
                        config,
                        root_interval=(float(eps_grid[left]), float(eps_grid[right])),
                    )
                    if fit is not None:
                        fits.append(fit)

                near = finite[np.abs(s_curve[finite]) < config.s_tol]
                if near.size > 0:
                    breaks = np.where(np.diff(near) > 1)[0] + 1
                    for segment in np.split(near, breaks):
                        if segment.size == 0:
                            continue
                        best = int(segment[np.argmin(np.abs(s_curve[segment]))])
                        fit = _refine_hopf_root_with_polynomial(
                            eps_grid,
                            omega_curve,
                            s_curve,
                            best,
                            float(eps_grid[best]),
                            n,
                            k,
                            mu,
                            ib,
                            config,
                        )
                        if fit is not None:
                            fits.append(fit)

    fits = _deduplicate_fits(fits, config)
    fits.sort(key=lambda p: (abs(p.s_value), p.n, p.k, p.epsilon, p.omega))
    return fits


def find_hopf_points(config: HopfSearchConfig = HopfSearchConfig()) -> list[HopfPoint]:
    return hopf_points_from_branches(compute_hopf_branches(config), config)


def plot_hopf_branches(
    branches: list[HopfBranchData],
    config: HopfSearchConfig = HopfSearchConfig(),
    *,
    show: bool = True,
    block: bool = False,
    polynomial_fits: list[HopfPolynomialFit] | None = None,
):
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if polynomial_fits is None:
        polynomial_fits = hopf_polynomial_fits_from_branches(branches, config)

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    for im, branch_data in enumerate(branches):
        color = colors[im % len(colors)]
        first = True
        for ib in range(branch_data.omega_mat.shape[1]):
            omega_curve = branch_data.omega_mat[:, ib]
            if np.all(~np.isfinite(omega_curve)):
                continue
            label = f"n={branch_data.n}" if first else None
            ax1.plot(branch_data.eps_grid, omega_curve, color=color, linewidth=1.2, label=label)
            first = False
    if polynomial_fits:
        ax1.scatter(
            [fit.epsilon for fit in polynomial_fits],
            [fit.omega for fit in polynomial_fits],
            marker="x",
            color="black",
            s=45,
            linewidths=1.4,
            label="polynomial root",
        )
    ax1.set_xlabel("epsilon")
    ax1.set_ylabel("omega")
    ax1.set_title("All root branches of G(omega, epsilon)=0")
    ax1.grid(True)
    ax1.legend(loc="best")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for im, branch_data in enumerate(branches):
        color = colors[im % len(colors)]
        for k in config.k_list:
            first = True
            for ib in range(branch_data.omega_mat.shape[1]):
                omega_curve = branch_data.omega_mat[:, ib]
                s_curve = _s_curve_for_branch(
                    branch_data.eps_grid,
                    omega_curve,
                    branch_data.mu,
                    k,
                    config,
                )
                if np.all(~np.isfinite(s_curve)):
                    continue
                label = f"n={branch_data.n}, k={k}" if first else None
                ax2.plot(branch_data.eps_grid, s_curve, color=color, linewidth=1.2, label=label)
                first = False
    for i, fit in enumerate(polynomial_fits):
        fit_label = "local polynomial fit" if i == 0 else None
        root_label = "polynomial root" if i == 0 else None
        sample_label = "fit samples" if i == 0 else None
        ax2.plot(
            fit.eps_plot,
            fit.s_plot,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=fit_label,
        )
        ax2.scatter(
            fit.eps_fit,
            fit.s_fit,
            color="white",
            edgecolor="black",
            s=18,
            linewidths=0.7,
            alpha=0.9,
            label=sample_label,
        )
        ax2.scatter(
            [fit.epsilon],
            [0.0],
            marker="x",
            color="black",
            s=45,
            linewidths=1.4,
            label=root_label,
        )
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("epsilon")
    ax2.set_ylabel("S_{n,k}")
    ax2.set_title("S_{n,k} values on root branches")
    ax2.grid(True)
    ax2.legend(loc="best")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    for im, branch_data in enumerate(branches):
        color = colors[im % len(colors)]
        for k in config.k_list:
            first = True
            for ib in range(branch_data.omega_mat.shape[1]):
                omega_curve = branch_data.omega_mat[:, ib]
                omega_keep = np.full(branch_data.eps_grid.shape, np.nan, dtype=float)
                s_curve = _s_curve_for_branch(
                    branch_data.eps_grid,
                    omega_curve,
                    branch_data.mu,
                    k,
                    config,
                )
                for ie, omega in enumerate(omega_curve):
                    if not np.isfinite(omega):
                        continue
                    sval = s_curve[ie]
                    if abs(sval) < config.s_tol:
                        omega_keep[ie] = omega
                if np.all(~np.isfinite(omega_keep)):
                    continue
                label = f"n={branch_data.n}, k={k}" if first else None
                ax3.plot(branch_data.eps_grid, omega_keep, color=color, linewidth=1.2, label=label)
                first = False
    if polynomial_fits:
        ax3.scatter(
            [fit.epsilon for fit in polynomial_fits],
            [fit.omega for fit in polynomial_fits],
            marker="x",
            color="black",
            s=45,
            linewidths=1.4,
            label="polynomial root",
        )
    ax3.set_xlabel("epsilon")
    ax3.set_ylabel("omega")
    ax3.set_title(f"Root candidates with local polynomial roots, |S_{{n,k}}| < {config.s_tol:g}")
    ax3.grid(True)
    ax3.legend(loc="best")

    for fig in (fig1, fig2, fig3):
        fig.tight_layout()
    if show and "agg" not in plt.get_backend().lower():
        plt.show(block=block)
        if not block:
            plt.pause(0.001)
    return fig1, fig2, fig3
