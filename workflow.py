from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import HopfRunConfig, RunConfig
from hopf_branch import (
    HopfBranchData,
    HopfPoint,
    HopfPolynomialFit,
    HopfSearchConfig,
    compute_hopf_branches,
    hopf_polynomial_fits_from_branches,
)
from io_utils import save_resolved_config, save_result_data
from plotting import (
    make_live_plot_callback,
    plot_final_result,
    render_hopf_plots,
    show_open_plots,
)
from reaction_diffusion_delay import (
    SimulationParams,
    make_cosine_initial_condition,
    run_simulation,
)


def run_workflow(config: RunConfig) -> None:
    output_dir = config.output.dir
    if config.output.save_resolved_config:
        save_resolved_config(config, output_dir / "resolved_config.json")
        print(f"Saved resolved config: {output_dir / 'resolved_config.json'}")

    base_params = config.model
    hopf_config = make_hopf_config(base_params, config.hopf)
    branches, polynomial_fits, points = find_hopf_candidates(hopf_config)

    saved_hopf_paths = render_hopf_plots(
        branches,
        hopf_config,
        config.plots,
        output_dir,
        polynomial_fits=polynomial_fits,
    )
    for path in saved_hopf_paths:
        print(f"Saved Hopf plot: {path}")

    if not points:
        raise RuntimeError("No Hopf branch candidates found.")

    print_hopf_summary(branches, points)

    if not config.simulation.enabled:
        print("Simulation disabled by configuration; Hopf search completed.")
        show_open_plots(config.plots)
        return

    selected_points = select_hopf_points(points, config.simulation.hopf_indices)
    if config.simulation.hopf_indices is None:
        print(f"No Hopf indices specified; simulating all {len(selected_points)} candidate(s).")

    for run_index, point in enumerate(selected_points):
        params = replace(base_params, epsilon=point.epsilon)
        mode = point.n
        print_selected_point(point, params)
        run_single_simulation(run_index, len(selected_points), params, mode, point, hopf_config, config)

    show_open_plots(config.plots)


def make_hopf_config(base_params: SimulationParams, hopf: HopfRunConfig) -> HopfSearchConfig:
    if not np.isclose(base_params.L / np.pi, round(base_params.L / np.pi), atol=1.0e-10):
        print(
            "Warning: Hopf theory uses mu=n^2/l^2 with L=l*pi. "
            f"Current L/pi={base_params.L / np.pi:.8f}; using this value as l."
        )

    return HopfSearchConfig(
        d1=base_params.d1,
        d2=base_params.d2,
        u_bar=base_params.u_bar,
        fu=base_params.r1 * (1.0 - 2.0 * base_params.u_bar),
        tau=base_params.tau,
        domain_factor=base_params.L / np.pi,
        n_list=hopf.n_list,
        k_list=hopf.k_list,
        eps_min=hopf.eps_min,
        eps_max_margin=hopf.eps_max_margin,
        omega_min=hopf.omega_min,
        omega_max=hopf.omega_max,
        num_eps=hopf.num_eps,
        num_omega=hopf.num_omega,
        min_hopf_omega=hopf.min_hopf_omega,
        max_roots_per_eps=hopf.max_roots_per_eps,
        max_total_branches=hopf.max_total_branches,
        branch_gap_abs=hopf.branch_gap_abs,
        branch_gap_rel=hopf.branch_gap_rel,
        branch_max_fill=hopf.branch_max_fill,
        s_tol=hopf.s_tol,
        polynomial_degree=hopf.polynomial_degree,
        polynomial_window=hopf.polynomial_window,
        polynomial_plot_points=hopf.polynomial_plot_points,
        duplicate_eps_tol=hopf.duplicate_eps_tol,
    )


def find_hopf_candidates(
    config: HopfSearchConfig,
) -> tuple[list[HopfBranchData], list[HopfPolynomialFit], list[HopfPoint]]:
    print(
        "Finding Hopf branch points with "
        f"num_eps={config.num_eps}, num_omega={config.num_omega}. "
        "Hopf model parameters are derived from SimulationParams: "
        f"d1={config.d1}, d2={config.d2}, u_bar={config.u_bar}, "
        f"fu={config.fu}, tau={config.tau}, l={config.domain_factor}."
    )
    branches = compute_hopf_branches(config)
    polynomial_fits = hopf_polynomial_fits_from_branches(branches, config)
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
        for fit in polynomial_fits
    ]
    points.sort(key=lambda p: (abs(p.s_value), p.n, p.k, p.epsilon, p.omega))
    return branches, polynomial_fits, points


def print_hopf_summary(branches: list[HopfBranchData], points: list[HopfPoint]) -> None:
    print("Hopf branch scan summary:")
    for branch in branches:
        root_count = int(np.count_nonzero(np.isfinite(branch.omega_mat)))
        branch_count = int(branch.omega_mat.shape[1])
        print(
            f"  n={branch.n}, mu={branch.mu:.8f}: "
            f"root_points={root_count}, branches={branch_count}"
        )

    print(f"Found {len(points)} Hopf candidate(s). Top candidates:")
    for i, point in enumerate(points[: min(8, len(points))]):
        print(
            f"  [{i}] epsilon={point.epsilon:.8f}, omega={point.omega:.8f}, "
            f"n={point.n}, k={point.k}, mu={point.mu:.8f}, S={point.s_value:.3e}"
        )


def select_hopf_points(
    points: list[HopfPoint],
    hopf_indices: tuple[int, ...] | None,
) -> list[HopfPoint]:
    if hopf_indices is None:
        return points

    max_index = len(points) - 1
    bad_indices = [index for index in hopf_indices if index > max_index]
    if bad_indices:
        raise ValueError(
            f"simulation.hopf_indices contains out-of-range value(s) {bad_indices}; "
            f"valid range is 0..{max_index}"
        )
    return [points[index] for index in hopf_indices]


def run_single_simulation(
    run_index: int,
    run_count: int,
    params: SimulationParams,
    mode: int,
    hopf_point: HopfPoint,
    hopf_config: HopfSearchConfig,
    config: RunConfig,
) -> None:
    amplitude = config.simulation.amplitude
    print(
        f"Simulation run {run_index + 1}/{run_count} parameters: "
        f"d1={params.d1}, d2={params.d2}, r1={params.r1}, "
        f"u_bar={params.u_bar}, tau={params.tau}, epsilon={params.epsilon}, "
        f"L={params.L}, Nx={params.Nx}, dt={params.dt}, Tend={params.Tend}, "
        f"initial_mode={mode}, amp={amplitude}"
    )

    run_title = format_run_title(
        params,
        mode,
        amplitude,
        hopf_point=hopf_point,
        hopf_config=hopf_config,
    )
    callback = (
        make_live_plot_callback(params, run_title)
        if config.plots.show and config.plots.live
        else None
    )

    result = run_simulation(
        params,
        initial_condition=make_cosine_initial_condition(
            params.u_bar,
            amplitude,
            mode,
            params.L,
        ),
        progress_callback=callback,
    )

    print(f"Finished run {run_index + 1}/{run_count}: steps={result.t.size - 1}, t={result.t[-1]:.3f}")
    print(
        "Final metrics: "
        f"min={result.min_u[-1]:.6f}, "
        f"max={result.max_u[-1]:.6f}, "
        f"mean={result.mean_u[-1]:.6f}, "
        f"max_cfl={np.max(result.cfl):.6e}, "
        f"last_residual={result.residuals[-1]:.6e}"
    )

    base_name = f"run_{run_index + 1:03d}"
    if config.output.save_data:
        data_path = config.output.dir / f"{base_name}_result.npz"
        save_result_data(
            result,
            params,
            mode,
            amplitude,
            data_path,
            hopf_point=hopf_point,
            hopf_config=hopf_config,
        )
        print(f"Saved result data: {data_path}")

    plot_path = plot_final_result(
        result,
        params,
        run_title,
        config.plots,
        save_path=config.output.dir / f"{base_name}_final.png",
    )
    if plot_path is not None:
        print(f"Saved final plot: {plot_path}")


def format_run_title(
    params: SimulationParams,
    mode: int,
    amplitude: float,
    *,
    hopf_point: HopfPoint,
    hopf_config: HopfSearchConfig,
) -> str:
    sim_line = (
        "Simulation: "
        f"d1={params.d1:g}, d2={params.d2:g}, r1={params.r1:g}, "
        f"u_bar={params.u_bar:g}, tau={params.tau:g}, epsilon={params.epsilon:.6g}, "
        f"L={params.L:.6g}, Nx={params.Nx}, dt={params.dt:g}, Tend={params.Tend:g}, "
        f"mode={mode}, amp={amplitude:g}"
    )
    hopf_line = (
        "Hopf: "
        f"epsilon={hopf_point.epsilon:.6g}, omega={hopf_point.omega:.6g}, "
        f"n={hopf_point.n}, k={hopf_point.k}, mu={hopf_point.mu:.6g}, "
        f"S={hopf_point.s_value:.3e}, "
        f"fu={hopf_config.fu:.6g}, l={hopf_config.domain_factor:.6g}, "
        f"eps_grid={hopf_config.num_eps}, omega_grid={hopf_config.num_omega}"
    )
    return sim_line + "\n" + hopf_line


def print_selected_point(point: HopfPoint, params: SimulationParams) -> None:
    print(
        "Selected Hopf point: "
        f"epsilon={point.epsilon:.8f}, omega={point.omega:.8f}, "
        f"n={point.n}, k={point.k}. "
        f"Simulation keeps d2={params.d2:.6g}, r1={params.r1:.6g}, L={params.L:.6g}; "
        "only epsilon is replaced by the selected branch value."
    )
