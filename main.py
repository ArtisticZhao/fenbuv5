from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from hopf_branch import (
    HopfSearchConfig,
    HopfPoint,
    compute_hopf_branches,
    hopf_points_from_branches,
    plot_hopf_branches,
)
from reaction_diffusion_delay import (
    SimulationParams,
    make_cosine_initial_condition,
    run_simulation,
)


MODEL_DEFAULTS = SimulationParams()
HOPF_DEFAULTS = HopfSearchConfig()


def format_int_list(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def format_run_title(
    params: SimulationParams,
    mode: int,
    amplitude: float,
    *,
    hopf_point: HopfPoint | None = None,
    hopf_config: HopfSearchConfig | None = None,
) -> str:
    sim_line = (
        "Simulation: "
        f"d1={params.d1:g}, d2={params.d2:g}, r1={params.r1:g}, "
        f"u_bar={params.u_bar:g}, tau={params.tau:g}, epsilon={params.epsilon:.6g}, "
        f"L={params.L:.6g}, Nx={params.Nx}, dt={params.dt:g}, Tend={params.Tend:g}, "
        f"mode={mode}, amp={amplitude:g}"
    )
    if hopf_point is None or hopf_config is None:
        return sim_line + "\nHopf: disabled"
    hopf_line = (
        "Hopf: "
        f"epsilon={hopf_point.epsilon:.6g}, omega={hopf_point.omega:.6g}, "
        f"n={hopf_point.n}, k={hopf_point.k}, mu={hopf_point.mu:.6g}, "
        f"S={hopf_point.s_value:.3e}, "
        f"fu={hopf_config.fu:.6g}, l={hopf_config.domain_factor:.6g}, "
        f"eps_grid={hopf_config.num_eps}, omega_grid={hopf_config.num_omega}"
    )
    return sim_line + "\n" + hopf_line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the delayed reaction-diffusion simulation."
    )
    parser.add_argument(
        "plot",
        nargs="?",
        type=int,
        default=None,
        help="1: update plots during simulation; 0: show plots after simulation.",
    )
    parser.add_argument(
        "--plot",
        dest="plot_kw",
        type=int,
        choices=(0, 1),
        default=None,
        help="1: update plots during simulation; 0: show plots after simulation.",
    )
    parser.add_argument(
        "--hopf",
        type=int,
        choices=(0, 1),
        default=1,
        help="1: find a Hopf branch point before simulation; 0: use fixed parameters.",
    )
    parser.add_argument(
        "--hopf-index",
        type=int,
        default=None,
        help="Index of the sorted Hopf candidate to simulate. Omit to simulate all candidates.",
    )
    parser.add_argument(
        "--plot-hopf",
        type=int,
        choices=(0, 1),
        default=1,
        help="1: draw Hopf branch-search figures before simulation.",
    )
    parser.add_argument(
        "--num-eps",
        type=int,
        default=HOPF_DEFAULTS.num_eps,
        help="Number of epsilon samples used by Hopf search.",
    )
    parser.add_argument(
        "--num-omega",
        type=int,
        default=HOPF_DEFAULTS.num_omega,
        help="Number of omega samples used by Hopf search.",
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=0.01,
        help="Initial perturbation amplitude.",
    )
    parser.add_argument(
        "--init-mode",
        type=int,
        default=2,
        help="Initial cosine mode used when --hopf 0.",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        choices=(0, 1),
        default=0,
        help="1: find parameters and print them without running the simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for saved final plots and numerical result files.",
    )
    parser.add_argument(
        "--tend",
        type=float,
        default=MODEL_DEFAULTS.Tend,
        help="Simulation end time.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=MODEL_DEFAULTS.dt,
        help="Simulation time step.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=MODEL_DEFAULTS.Nx,
        help="Number of spatial cells.",
    )
    parser.add_argument("--d1", type=float, default=MODEL_DEFAULTS.d1, help="Diffusion coefficient d1.")
    parser.add_argument("--d2", type=float, default=MODEL_DEFAULTS.d2, help="Nonlocal flux coefficient d2.")
    parser.add_argument("--r1", type=float, default=MODEL_DEFAULTS.r1, help="Logistic reaction coefficient r1.")
    parser.add_argument("--u-bar", type=float, default=MODEL_DEFAULTS.u_bar, help="Homogeneous steady state u_bar.")
    parser.add_argument("--tau", type=float, default=MODEL_DEFAULTS.tau, help="Delay length tau.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=MODEL_DEFAULTS.epsilon,
        help="Delay kernel epsilon used when --hopf 0. Hopf mode replaces it with the selected branch value.",
    )
    parser.add_argument("--L", type=float, default=MODEL_DEFAULTS.L, help="Domain length L.")
    parser.add_argument(
        "--sweep-tol",
        type=float,
        default=MODEL_DEFAULTS.sweep_tol,
        help="FiPy sweep residual tolerance.",
    )
    parser.add_argument(
        "--stagnation-tol",
        type=float,
        default=MODEL_DEFAULTS.stagnation_tol,
        help="Sweep stagnation tolerance.",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=MODEL_DEFAULTS.max_sweeps,
        help="Maximum sweeps per time step.",
    )
    parser.add_argument(
        "--omega-min",
        type=float,
        default=HOPF_DEFAULTS.omega_min,
        help="Minimum omega used by Hopf search.",
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        default=HOPF_DEFAULTS.omega_max,
        help="Maximum omega used by Hopf search.",
    )
    parser.add_argument(
        "--eps-min",
        type=float,
        default=HOPF_DEFAULTS.eps_min,
        help="Minimum epsilon used by Hopf search.",
    )
    parser.add_argument(
        "--eps-max-margin",
        type=float,
        default=HOPF_DEFAULTS.eps_max_margin,
        help="Hopf search uses tau - eps_max_margin as maximum epsilon.",
    )
    parser.add_argument(
        "--n-list",
        type=str,
        default=format_int_list(HOPF_DEFAULTS.n_list),
        help="Comma-separated mode list used by Hopf search.",
    )
    parser.add_argument(
        "--k-list",
        type=str,
        default=format_int_list(HOPF_DEFAULTS.k_list),
        help="Comma-separated k list used by Hopf search.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not items:
        raise ValueError("integer list must not be empty")
    return items


def make_model_params(args: argparse.Namespace) -> SimulationParams:
    return SimulationParams(
        d1=args.d1,
        d2=args.d2,
        r1=args.r1,
        u_bar=args.u_bar,
        tau=args.tau,
        epsilon=args.epsilon,
        L=args.L,
        Nx=args.nx,
        dt=args.dt,
        Tend=args.tend,
        sweep_tol=args.sweep_tol,
        stagnation_tol=args.stagnation_tol,
        max_sweeps=args.max_sweeps,
    )


def find_hopf_simulation_params(
    base_params: SimulationParams,
    candidate_index: int | None,
    num_eps: int,
    num_omega: int,
    eps_min: float,
    eps_max_margin: float,
    omega_min: float,
    omega_max: float,
    n_list: tuple[int, ...],
    k_list: tuple[int, ...],
    plot_hopf: bool,
    block_hopf_plot: bool = False,
) -> list[tuple[SimulationParams, int, HopfPoint, HopfSearchConfig]]:
    if base_params.L <= 0:
        raise ValueError("L must be positive")
    if not np.isclose(base_params.L / np.pi, round(base_params.L / np.pi), atol=1.0e-10):
        print(
            "Warning: Hopf theory uses mu=n^2/l^2 with L=l*pi. "
            f"Current L/pi={base_params.L / np.pi:.8f}; using this value as l."
        )

    config = HopfSearchConfig(
        d1=base_params.d1,
        d2=base_params.d2,
        u_bar=base_params.u_bar,
        fu=base_params.r1 * (1.0 - 2.0 * base_params.u_bar),
        tau=base_params.tau,
        domain_factor=base_params.L / np.pi,
        n_list=n_list,
        k_list=k_list,
        eps_min=eps_min,
        eps_max_margin=eps_max_margin,
        omega_min=omega_min,
        omega_max=omega_max,
        num_eps=num_eps,
        num_omega=num_omega,
    )
    print(
        "Finding Hopf branch points with "
        f"num_eps={config.num_eps}, num_omega={config.num_omega}. "
        "Hopf model parameters are derived from SimulationParams: "
        f"d1={config.d1}, d2={config.d2}, u_bar={config.u_bar}, "
        f"fu={config.fu}, tau={config.tau}, l={config.domain_factor}."
    )
    branches = compute_hopf_branches(config)
    points = hopf_points_from_branches(branches, config)

    print("Hopf branch scan summary:")
    for branch in branches:
        root_count = int(np.count_nonzero(np.isfinite(branch.omega_mat)))
        branch_count = int(branch.omega_mat.shape[1])
        print(
            f"  n={branch.n}, mu={branch.mu:.8f}: "
            f"root_points={root_count}, branches={branch_count}"
        )

    if not points:
        if plot_hopf:
            plot_hopf_branches(branches, config, show=True, block=block_hopf_plot)
        raise RuntimeError("No Hopf branch candidates found.")
    if candidate_index is not None and not 0 <= candidate_index < len(points):
        if plot_hopf:
            plot_hopf_branches(branches, config, show=True, block=block_hopf_plot)
        raise ValueError(
            f"hopf-index must be between 0 and {len(points) - 1}; got {candidate_index}"
        )

    print(f"Found {len(points)} Hopf candidate(s). Top candidates:")
    for i, point in enumerate(points[: min(8, len(points))]):
        print(
            f"  [{i}] epsilon={point.epsilon:.8f}, omega={point.omega:.8f}, "
            f"n={point.n}, k={point.k}, mu={point.mu:.8f}, S={point.s_value:.3e}"
        )

    if plot_hopf:
        plot_hopf_branches(branches, config, show=True, block=block_hopf_plot)

    selected_points = points if candidate_index is None else [points[candidate_index]]
    if candidate_index is None:
        print(f"No hopf-index specified; simulating all {len(selected_points)} candidate(s).")

    run_specs = []
    for point in selected_points:
        params = replace(base_params, epsilon=point.epsilon)
        print(
            "Selected Hopf point: "
            f"epsilon={point.epsilon:.8f}, omega={point.omega:.8f}, "
            f"n={point.n}, k={point.k}. "
            f"Simulation keeps d2={params.d2:.6g}, r1={params.r1:.6g}, L={params.L:.6g}; "
            "only epsilon is replaced by the selected branch value."
        )
        run_specs.append((params, point.n, point, config))
    return run_specs


def make_live_plot_callback(
    params: SimulationParams,
    title: str,
    update_step: int = 200,
):
    steps = int(round(params.Tend / params.dt))
    solution = np.zeros((steps + 1, params.Nx), dtype=float)
    residuals = np.zeros(steps + 1, dtype=float)
    probe_trace = np.zeros((3, steps + 1), dtype=float)
    time = np.linspace(0.0, steps * params.dt, steps + 1)
    probe_indices = None
    probe_positions = np.array([params.L / 4.0, params.L / 2.0, 3.0 * params.L / 4.0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=10)
    ax1, ax2, ax3, ax4 = axes.ravel()

    line, = ax1.plot([], [], "b-", linewidth=2)
    ax1.set_ylim(0, 3)
    ax1.set_xlim(0, params.L)
    ax1.set_ylabel("u(x, t)")
    ax1.set_xlabel("Space (x)")
    ax1.set_title("u(x) Current State")
    ax1.grid(True)

    heatmap = ax2.imshow(
        solution,
        aspect="auto",
        cmap="jet",
        vmin=0,
        vmax=2.5,
        extent=[0, params.L, 0, steps * params.dt],
        origin="lower",
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_title("u(x,t)")

    resid_line, = ax3.plot(np.arange(steps + 1), residuals)
    ax3.set_ylim(0, 1.0e-3)
    ax3.set_xlim(0, steps)
    ax3.set_ylabel("Residual")
    ax3.set_xlabel("Step")
    ax3.set_title("Residual of sweep")
    ax3.grid(True)

    probe_lines = [
        ax4.plot(time, probe_trace[i], linewidth=1.5, label=label)[0]
        for i, label in enumerate(("L/4", "L/2", "3L/4"))
    ]
    ax4.set_xlim(0, steps * params.dt)
    ax4.set_ylim(0, 3)
    ax4.set_xlabel("t")
    ax4.set_ylabel("u(x_probe, t)")
    ax4.set_title("Probe Time Series")
    ax4.legend(loc="best")
    ax4.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show(block=False)

    def callback(state: dict[str, object]) -> None:
        nonlocal probe_indices
        step = int(state["step"])
        x = np.asarray(state["x"], dtype=float)
        u = np.asarray(state["u"], dtype=float)
        residual = float(state["residual"])

        if probe_indices is None:
            probe_indices = [
                int(np.argmin(np.abs(x - position))) for position in probe_positions
            ]

        solution[step, :] = u
        probe_trace[:, step] = u[probe_indices]
        if np.isfinite(residual):
            residuals[step] = residual

        if step % update_step != 0 and step != steps:
            return

        line.set_data(x, u)
        heatmap.set_data(solution)
        resid_line.set_ydata(residuals)
        for i, probe_line in enumerate(probe_lines):
            probe_line.set_ydata(probe_trace[i])
        ax1.set_title(f"u(x), t = {float(state['time']):.2f}")
        fig.canvas.draw_idle()
        plt.pause(0.001)

    return callback


def save_result_data(
    result,
    params: SimulationParams,
    mode: int,
    amplitude: float,
    output_path: Path,
    *,
    hopf_point: HopfPoint | None = None,
    hopf_config: HopfSearchConfig | None = None,
) -> None:
    hopf_values = {
        "hopf_enabled": hopf_point is not None,
        "hopf_epsilon": np.nan,
        "hopf_omega": np.nan,
        "hopf_n": -1,
        "hopf_k": -1,
        "hopf_mu": np.nan,
        "hopf_s_value": np.nan,
        "hopf_fu": np.nan,
        "hopf_domain_factor": np.nan,
    }
    if hopf_point is not None and hopf_config is not None:
        hopf_values.update(
            {
                "hopf_epsilon": hopf_point.epsilon,
                "hopf_omega": hopf_point.omega,
                "hopf_n": hopf_point.n,
                "hopf_k": hopf_point.k,
                "hopf_mu": hopf_point.mu,
                "hopf_s_value": hopf_point.s_value,
                "hopf_fu": hopf_config.fu,
                "hopf_domain_factor": hopf_config.domain_factor,
            }
        )
    np.savez_compressed(
        output_path,
        x=result.x,
        t=result.t,
        u=result.u,
        residuals=result.residuals,
        sweep_counts=result.sweep_counts,
        min_u=result.min_u,
        max_u=result.max_u,
        mean_u=result.mean_u,
        cfl=result.cfl,
        h_weights=result.h_weights,
        s_vec=result.s_vec,
        d1=params.d1,
        d2=params.d2,
        r1=params.r1,
        u_bar=params.u_bar,
        tau=params.tau,
        epsilon=params.epsilon,
        L=params.L,
        Nx=params.Nx,
        dt=params.dt,
        Tend=params.Tend,
        sweep_tol=params.sweep_tol,
        stagnation_tol=params.stagnation_tol,
        max_sweeps=params.max_sweeps,
        initial_mode=mode,
        amplitude=amplitude,
        **hopf_values,
    )


def plot_final_result(
    result,
    params: SimulationParams,
    title: str,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    probe_positions = np.array([params.L / 4.0, params.L / 2.0, 3.0 * params.L / 4.0])
    probe_indices = [
        int(np.argmin(np.abs(result.x - position))) for position in probe_positions
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=10)
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(result.x, result.u[-1], "b-", linewidth=2)
    ax1.set_ylim(0, 3)
    ax1.set_xlabel("Space (x)")
    ax1.set_ylabel("u(x, t)")
    ax1.set_title(f"Final State, t = {result.t[-1]:.2f}")
    ax1.grid(True)

    heatmap = ax2.imshow(
        result.u,
        aspect="auto",
        cmap="jet",
        vmin=0,
        vmax=2.5,
        extent=[0, params.L, 0, result.t[-1]],
        origin="lower",
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_title("u(x,t)")
    plt.colorbar(heatmap, ax=ax2, label="Concentration u")

    ax3.plot(np.arange(result.residuals.size), result.residuals)
    ax3.set_ylim(0, 1.0e-3)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Residual")
    ax3.set_title("Residual of sweep")
    ax3.grid(True)

    for index, label in zip(probe_indices, ("L/4", "L/2", "3L/4")):
        ax4.plot(result.t, result.u[:, index], linewidth=1.5, label=label)
    ax4.set_xlabel("t")
    ax4.set_ylabel("u(x_probe, t)")
    ax4.set_title("Probe Time Series")
    ax4.legend(loc="best")
    ax4.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    plot = args.plot_kw if args.plot_kw is not None else args.plot
    if plot is None:
        plot = 0
    if plot not in (0, 1):
        raise ValueError("plot must be 0 or 1")

    base_params = make_model_params(args)
    if args.hopf == 1:
        run_specs = find_hopf_simulation_params(
            base_params,
            args.hopf_index,
            args.num_eps,
            args.num_omega,
            args.eps_min,
            args.eps_max_margin,
            args.omega_min,
            args.omega_max,
            parse_int_list(args.n_list),
            parse_int_list(args.k_list),
            args.plot_hopf == 1,
            args.dry_run == 1,
        )
    else:
        run_specs = [(base_params, args.init_mode, None, None)]

    if args.dry_run != 1:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for run_index, (params, mode, hopf_point, hopf_config) in enumerate(run_specs):
        print(
            f"Simulation run {run_index + 1}/{len(run_specs)} parameters: "
            f"d1={params.d1}, d2={params.d2}, r1={params.r1}, "
            f"u_bar={params.u_bar}, tau={params.tau}, epsilon={params.epsilon}, "
            f"L={params.L}, Nx={params.Nx}, dt={params.dt}, Tend={params.Tend}, "
            f"initial_mode={mode}, amp={args.amp}"
        )
        if args.dry_run == 1:
            continue

        run_title = format_run_title(
            params,
            mode,
            args.amp,
            hopf_point=hopf_point,
            hopf_config=hopf_config,
        )
        callback = make_live_plot_callback(params, run_title) if plot == 1 else None

        result = run_simulation(
            params,
            initial_condition=make_cosine_initial_condition(
                params.u_bar,
                args.amp,
                mode,
                params.L,
            ),
            progress_callback=callback,
        )

        print(f"Finished run {run_index + 1}/{len(run_specs)}: steps={result.t.size - 1}, t={result.t[-1]:.3f}")
        print(
            "Final metrics: "
            f"min={result.min_u[-1]:.6f}, "
            f"max={result.max_u[-1]:.6f}, "
            f"mean={result.mean_u[-1]:.6f}, "
            f"max_cfl={np.max(result.cfl):.6e}, "
            f"last_residual={result.residuals[-1]:.6e}"
        )

        base_name = f"run_{run_index + 1:03d}"
        plot_path = args.output_dir / f"{base_name}_final.png"
        data_path = args.output_dir / f"{base_name}_result.npz"
        save_result_data(
            result,
            params,
            mode,
            args.amp,
            data_path,
            hopf_point=hopf_point,
            hopf_config=hopf_config,
        )
        print(f"Saved result data: {data_path}")

        if plot == 0:
            plot_final_result(result, params, run_title, save_path=plot_path, show=True)
        else:
            plot_final_result(result, params, run_title, save_path=plot_path, show=False)
        print(f"Saved final plot: {plot_path}")

    if args.dry_run == 1:
        return
    if plot == 1:
        plt.show()


if __name__ == "__main__":
    main()
