from __future__ import annotations

import numpy as np

from reaction_diffusion_delay import (
    SimulationParams,
    delay_average,
    heat_equation_exact,
    l2_error,
    logistic_exact,
    make_cosine_initial_condition,
    run_simulation,
    triangular_delay_weights,
)


def assert_close(name: str, value: float, limit: float) -> None:
    if not value <= limit:
        raise AssertionError(f"{name}: {value:.6e} > {limit:.6e}")
    print(f"PASS {name}: {value:.6e} <= {limit:.6e}")


def assert_small_range(name: str, values: np.ndarray, center: float, limit: float) -> None:
    err = float(np.max(np.abs(values - center)))
    assert_close(name, err, limit)


def test_delay_weights_constant_history() -> None:
    tau = 2.0
    epsilon = 1.5
    dt = 0.005
    _, weights = triangular_delay_weights(tau, epsilon, dt)
    weight_sum_error = abs(float(np.sum(weights)) - 1.0)
    assert_close("delay kernel normalization", weight_sum_error, 5.0e-3)

    constant = 2.75
    history = [np.full(11, constant) for _ in range(weights.size)]
    averaged = delay_average(history, weights)
    assert_small_range("constant history delay average", averaged, constant, 2.0e-2)


def test_delay_weights_require_dt_alignment() -> None:
    try:
        triangular_delay_weights(tau=2.0, epsilon=1.5, dt=0.03)
    except ValueError as exc:
        if "integer multiple" not in str(exc):
            raise AssertionError(f"unexpected error message: {exc}") from exc
    else:
        raise AssertionError("expected non-aligned tau/dt to raise ValueError")
    print("PASS delay grid rejects non-aligned tau/dt")


def test_constant_steady_state() -> None:
    params = SimulationParams(
        d1=1.0,
        d2=3.0,
        r1=0.1,
        tau=2.0,
        epsilon=1.5,
        L=np.pi,
        Nx=51,
        dt=0.05,
        Tend=2.0,
    )
    result = run_simulation(params, initial_condition=lambda x: np.ones_like(x))
    err = l2_error(result.u[-1], np.ones_like(result.x))
    assert_close("u=1 steady state", err, 2.0e-10)


def test_heat_equation_mode_decay() -> None:
    u_bar = 1.0
    amplitude = 0.01
    mode = 2
    params = SimulationParams(
        d1=1.0,
        d2=0.0,
        r1=0.0,
        tau=2.0,
        epsilon=1.5,
        L=np.pi,
        Nx=201,
        dt=0.001,
        Tend=0.05,
        sweep_tol=1.0e-10,
        stagnation_tol=1.0e-12,
    )
    result = run_simulation(
        params,
        initial_condition=make_cosine_initial_condition(u_bar, amplitude, mode, params.L),
    )
    exact = heat_equation_exact(
        result.x,
        result.t[-1],
        d1=params.d1,
        L=params.L,
        u_bar=u_bar,
        amplitude=amplitude,
        mode=mode,
    )
    err = l2_error(result.u[-1], exact)
    assert_close("heat equation cosine decay", err, 3.0e-5)


def test_logistic_reaction_against_exact_solution() -> None:
    params = SimulationParams(
        d1=0.0,
        d2=0.0,
        r1=0.1,
        tau=2.0,
        epsilon=1.5,
        L=np.pi,
        Nx=51,
        dt=0.001,
        Tend=0.05,
        sweep_tol=1.0e-6,
        stagnation_tol=1.0e-14,
    )
    initial = lambda x: 0.8 + 0.01 * np.cos(2.0 * np.pi * x / params.L)
    result = run_simulation(params, initial_condition=initial)
    u0 = initial(result.x)
    exact = logistic_exact(u0, result.t[-1], params.r1)
    err = l2_error(result.u[-1], exact)
    assert_close("logistic reaction exact solution", err, 2.0e-5)


def test_short_full_model_sanity() -> None:
    params = SimulationParams(
        d1=1.0,
        d2=3.0,
        r1=0.1,
        tau=2.0,
        epsilon=1.5,
        L=np.pi,
        Nx=51,
        dt=0.01,
        Tend=1.0,
    )
    initial = lambda x: 0.8 + 0.01 * np.cos(2.0 * np.pi * x / params.L)
    result = run_simulation(params, initial_condition=initial)

    if not np.all(np.isfinite(result.u)):
        raise AssertionError("full model produced non-finite values")
    if result.min_u[-1] <= 0.0:
        raise AssertionError(f"full model produced non-positive u: min={result.min_u[-1]}")
    if result.max_u[-1] >= 2.0:
        raise AssertionError(f"full model produced unexpectedly large u: max={result.max_u[-1]}")
    if np.nanmax(result.residuals[1:]) > 1.0e-2:
        raise AssertionError(f"large solver residual: max={np.nanmax(result.residuals[1:])}")

    print(f"PASS full model finite: min={result.min_u[-1]:.6f}, max={result.max_u[-1]:.6f}")
    print(f"INFO full model max CFL: {np.max(result.cfl):.6e}")


def main() -> None:
    tests = [
        test_delay_weights_constant_history,
        test_delay_weights_require_dt_alignment,
        test_constant_steady_state,
        test_heat_equation_mode_decay,
        test_logistic_reaction_against_exact_solution,
        test_short_full_model_sanity,
    ]
    for test in tests:
        print(f"\n== {test.__name__} ==")
        test()


if __name__ == "__main__":
    main()
