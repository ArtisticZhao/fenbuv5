from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from fipy import (
    CellVariable,
    DiffusionTerm,
    Grid1D,
    ImplicitSourceTerm,
    TransientTerm,
    UpwindConvectionTerm,
)


ArrayLikeFn = Callable[[np.ndarray], np.ndarray]
ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class SimulationParams:
    d1: float = 1.0
    d2: float = 3.0
    r1: float = 0.1
    tau: float = 2.0
    epsilon: float = 1.5
    L: float = np.pi
    Nx: int = 101
    dt: float = 0.05
    Tend: float = 300.0
    sweep_tol: float = 1.0e-4
    stagnation_tol: float = 1.0e-5
    max_sweeps: int = 200


@dataclass
class SimulationResult:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    residuals: np.ndarray
    sweep_counts: np.ndarray
    min_u: np.ndarray
    max_u: np.ndarray
    mean_u: np.ndarray
    cfl: np.ndarray
    h_weights: np.ndarray
    s_vec: np.ndarray


def triangular_delay_weights(
    tau: float,
    epsilon: float,
    dt: float,
    *,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return discrete weights for the piecewise triangular delay kernel."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not 0 < epsilon < tau:
        raise ValueError("epsilon must satisfy 0 < epsilon < tau")
    if dt <= 0:
        raise ValueError("dt must be positive")

    history_steps = int(np.round(tau / dt)) + 1
    if history_steps % 2 == 0:
        history_steps += 1

    s_vec = np.linspace(-tau, 0.0, history_steps)
    h = np.zeros(history_steps, dtype=float)

    left = (-tau <= s_vec) & (s_vec <= -epsilon)
    right = (-epsilon <= s_vec) & (s_vec <= 0.0)
    h[left] = 2.0 / (tau * (tau - epsilon)) * (s_vec[left] + tau)
    h[right] = -2.0 / (tau * epsilon) * s_vec[right]

    ds = s_vec[1] - s_vec[0]
    weights = h * ds
    if normalize:
        total = float(np.sum(weights))
        if total == 0:
            raise ValueError("delay weights sum to zero")
        weights = weights / total
    return s_vec, weights


def delay_average(history: deque[np.ndarray], weights: np.ndarray) -> np.ndarray:
    history_matrix = np.asarray(history)
    if history_matrix.shape[0] != weights.size:
        raise ValueError("history length and weights length differ")
    return np.dot(weights, history_matrix)


def make_cosine_initial_condition(
    u_bar: float = 1.0,
    amplitude: float = 0.01,
    mode: int = 2,
    L: float = np.pi,
) -> ArrayLikeFn:
    def initial_condition(x: np.ndarray) -> np.ndarray:
        return u_bar + amplitude * np.cos(mode * np.pi * x / L)

    return initial_condition


def heat_equation_exact(
    x: np.ndarray,
    t: float,
    *,
    d1: float,
    L: float,
    u_bar: float,
    amplitude: float,
    mode: int,
) -> np.ndarray:
    lam = d1 * (mode * np.pi / L) ** 2
    return u_bar + amplitude * np.exp(-lam * t) * np.cos(mode * np.pi * x / L)


def logistic_exact(u0: np.ndarray, t: float, r1: float) -> np.ndarray:
    if r1 == 0:
        return np.asarray(u0, dtype=float).copy()
    exp_rt = np.exp(r1 * t)
    return u0 * exp_rt / (1.0 + u0 * (exp_rt - 1.0))


def run_simulation(
    params: SimulationParams,
    initial_condition: ArrayLikeFn | None = None,
    *,
    normalize_delay_weights: bool = False,
    store_history: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> SimulationResult:
    """Run the 1D delayed reaction-diffusion simulation."""
    if params.Nx < 3:
        raise ValueError("Nx must be at least 3")
    if params.dt <= 0 or params.Tend < 0:
        raise ValueError("dt must be positive and Tend must be non-negative")

    steps = int(round(params.Tend / params.dt))
    mesh = Grid1D(nx=params.Nx, Lx=params.L)
    x = np.asarray(mesh.cellCenters[0], dtype=float)

    if initial_condition is None:
        initial_condition = make_cosine_initial_condition(
            u_bar=params.u_bar,
            amplitude=0.01,
            mode=2,
            L=params.L,
        )

    u = CellVariable(name="u", mesh=mesh, value=0.0, hasOld=True)
    u_tau = CellVariable(name="u_tau", mesh=mesh, value=0.0)
    u.setValue(initial_condition(x))
    u.updateOld()

    s_vec, h_weights = triangular_delay_weights(
        params.tau,
        params.epsilon,
        params.dt,
        normalize=normalize_delay_weights,
    )
    u_history: deque[np.ndarray] = deque(
        [u.value.copy() for _ in range(h_weights.size)],
        maxlen=h_weights.size,
    )

    rhs = None
    velocity = None
    if params.d1 != 0:
        rhs = DiffusionTerm(coeff=params.d1, var=u)
    if params.d2 != 0:
        velocity = params.d2 * u_tau.faceGrad
        advection = UpwindConvectionTerm(coeff=velocity, var=u)
        rhs = advection if rhs is None else rhs + advection
    if params.r1 != 0:
        reaction_explicit = params.r1 * u
        reaction_implicit = ImplicitSourceTerm(coeff=-params.r1 * u, var=u)
        reaction = reaction_explicit + reaction_implicit
        rhs = reaction if rhs is None else rhs + reaction
    if rhs is None:
        rhs = 0
    eq = TransientTerm(var=u) == rhs

    nt = steps + 1
    saved_u = np.zeros((nt, params.Nx), dtype=float) if store_history else np.zeros((1, params.Nx), dtype=float)
    saved_u[0, :] = u.value
    residuals = np.full(nt, np.nan, dtype=float)
    sweep_counts = np.zeros(nt, dtype=int)
    min_u = np.empty(nt, dtype=float)
    max_u = np.empty(nt, dtype=float)
    mean_u = np.empty(nt, dtype=float)
    cfl = np.zeros(nt, dtype=float)

    min_u[0] = float(np.min(u.value))
    max_u[0] = float(np.max(u.value))
    mean_u[0] = float(np.mean(u.value))

    if progress_callback is not None:
        progress_callback(
            {
                "step": 0,
                "time": 0.0,
                "x": x,
                "u": u.value.copy(),
                "residual": np.nan,
                "sweep_count": 0,
                "min_u": min_u[0],
                "max_u": max_u[0],
                "mean_u": mean_u[0],
                "cfl": cfl[0],
            }
        )

    dx = params.L / params.Nx

    for step in range(1, nt):
        u_tau_val = delay_average(u_history, h_weights)
        u_tau.setValue(u_tau_val)

        if velocity is not None:
            velocity_value = np.asarray(velocity.value, dtype=float)
            cfl[step] = float(np.max(np.abs(velocity_value)) * params.dt / dx)

        resid = np.inf
        last_resid = np.inf
        count = 0
        while resid > params.sweep_tol and count < params.max_sweeps:
            resid = float(eq.sweep(var=u, dt=params.dt))
            count += 1
            if abs(resid - last_resid) < params.stagnation_tol:
                break
            last_resid = resid

        u_history.append(u.value.copy())
        if store_history:
            saved_u[step, :] = u.value
        residuals[step] = resid
        sweep_counts[step] = count
        min_u[step] = float(np.min(u.value))
        max_u[step] = float(np.max(u.value))
        mean_u[step] = float(np.mean(u.value))
        u.updateOld()

        if progress_callback is not None:
            progress_callback(
                {
                    "step": step,
                    "time": step * params.dt,
                    "x": x,
                    "u": u.value.copy(),
                    "residual": residuals[step],
                    "sweep_count": sweep_counts[step],
                    "min_u": min_u[step],
                    "max_u": max_u[step],
                    "mean_u": mean_u[step],
                    "cfl": cfl[step],
                }
            )

    t = np.linspace(0.0, steps * params.dt, nt)
    return SimulationResult(
        x=x,
        t=t,
        u=saved_u,
        residuals=residuals,
        sweep_counts=sweep_counts,
        min_u=min_u,
        max_u=max_u,
        mean_u=mean_u,
        cfl=cfl,
        h_weights=h_weights,
        s_vec=s_vec,
    )


def l2_error(numerical: np.ndarray, exact: np.ndarray) -> float:
    diff = np.asarray(numerical) - np.asarray(exact)
    return float(np.sqrt(np.mean(diff * diff)))
