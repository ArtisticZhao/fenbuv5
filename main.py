from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from hopf_branch import (
    HopfSearchConfig,
    compute_hopf_branches,
    hopf_points_from_branches,
    plot_hopf_branches,
)
from reaction_diffusion_delay import SimulationParams, run_simulation


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
        default=0,
        help="Index of the sorted Hopf candidate to simulate.",
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
        default=220,
        help="Number of epsilon samples used by Hopf search.",
    )
    parser.add_argument(
        "--num-omega",
        type=int,
        default=12000,
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
        "--tend",
        type=float,
        default=None,
        help="Override simulation end time.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override simulation time step.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=None,
        help="Override number of spatial cells.",
    )
    parser.add_argument("--d1", type=float, default=1.0, help="Diffusion coefficient d1.")
    parser.add_argument("--d2", type=float, default=-5.8, help="Nonlocal flux coefficient d2.")
    parser.add_argument("--r1", type=float, default=0.5, help="Logistic reaction coefficient r1.")
    parser.add_argument("--u-bar", type=float, default=1.0, help="Homogeneous steady state u_bar.")
    parser.add_argument("--tau", type=float, default=2.0, help="Delay length tau.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.8,
        help="Delay kernel epsilon used when --hopf 0. Hopf mode replaces it with the selected branch value.",
    )
    parser.add_argument("--L", type=float, default=3.0 * np.pi, help="Domain length L.")
    parser.add_argument(
        "--sweep-tol",
        type=float,
        default=1.0e-4,
        help="FiPy sweep residual tolerance.",
    )
    parser.add_argument(
        "--stagnation-tol",
        type=float,
        default=1.0e-5,
        help="Sweep stagnation tolerance.",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=200,
        help="Maximum sweeps per time step.",
    )
    parser.add_argument(
        "--omega-min",
        type=float,
        default=1.0e-4,
        help="Minimum omega used by Hopf search.",
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        default=80.0,
        help="Maximum omega used by Hopf search.",
    )
    parser.add_argument(
        "--eps-min",
        type=float,
        default=0.02,
        help="Minimum epsilon used by Hopf search.",
    )
    parser.add_argument(
        "--eps-max-margin",
        type=float,
        default=0.02,
        help="Hopf search uses tau - eps_max_margin as maximum epsilon.",
    )
    parser.add_argument(
        "--n-list",
        type=str,
        default="1,3,5,7",
        help="Comma-separated mode list used by Hopf search.",
    )
    parser.add_argument(
        "--k-list",
        type=str,
        default="0",
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
        Nx=args.nx if args.nx is not None else 101,
        dt=args.dt if args.dt is not None else 0.05,
        Tend=args.tend if args.tend is not None else 300.0,
        sweep_tol=args.sweep_tol,
        stagnation_tol=args.stagnation_tol,
        max_sweeps=args.max_sweeps,
    )


def make_initial_condition(params: SimulationParams, mode: int, amplitude: float):
    def initial_condition(x: np.ndarray) -> np.ndarray:
        return params.u_bar + amplitude * np.cos(mode * np.pi * x / params.L)

    return initial_condition


def find_hopf_simulation_params(
    base_params: SimulationParams,
    candidate_index: int,
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
) -> tuple[SimulationParams, int]:
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
    if not 0 <= candidate_index < len(points):
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

    point = points[candidate_index]
    if plot_hopf:
        plot_hopf_branches(branches, config, show=True, block=block_hopf_plot)

    params = SimulationParams(
        d1=base_params.d1,
        d2=base_params.d2,
        r1=base_params.r1,
        u_bar=base_params.u_bar,
        tau=base_params.tau,
        epsilon=point.epsilon,
        L=base_params.L,
        Nx=base_params.Nx,
        dt=base_params.dt,
        Tend=base_params.Tend,
        sweep_tol=base_params.sweep_tol,
        stagnation_tol=base_params.stagnation_tol,
        max_sweeps=base_params.max_sweeps,
    )
    print(
        "Selected Hopf point: "
        f"epsilon={point.epsilon:.8f}, omega={point.omega:.8f}, "
        f"n={point.n}, k={point.k}. "
        f"Simulation keeps d2={params.d2:.6g}, r1={params.r1:.6g}, L={params.L:.6g}; "
        "only epsilon is replaced by the selected branch value."
    )
    return params, point.n


def make_live_plot_callback(params: SimulationParams, update_step: int = 200):
    steps = int(round(params.Tend / params.dt))
    solution = np.zeros((steps + 1, params.Nx), dtype=float)
    residuals = np.zeros(steps + 1, dtype=float)
    probe_trace = np.zeros((3, steps + 1), dtype=float)
    time = np.linspace(0.0, steps * params.dt, steps + 1)
    probe_indices = None
    probe_positions = np.array([params.L / 4.0, params.L / 2.0, 3.0 * params.L / 4.0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
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

    plt.tight_layout()
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


def plot_final_result(result, params: SimulationParams) -> None:
    probe_positions = np.array([params.L / 4.0, params.L / 2.0, 3.0 * params.L / 4.0])
    probe_indices = [
        int(np.argmin(np.abs(result.x - position))) for position in probe_positions
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
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

    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    plot = args.plot_kw if args.plot_kw is not None else args.plot
    if plot is None:
        plot = 0
    if plot not in (0, 1):
        raise ValueError("plot must be 0 or 1")

    base_params = make_model_params(args)
    if args.hopf == 1:
        params, mode = find_hopf_simulation_params(
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
        params = base_params
        mode = args.init_mode

    print(
        "Simulation parameters: "
        f"d1={params.d1}, d2={params.d2}, r1={params.r1}, "
        f"u_bar={params.u_bar}, tau={params.tau}, epsilon={params.epsilon}, "
        f"L={params.L}, Nx={params.Nx}, dt={params.dt}, Tend={params.Tend}, "
        f"initial_mode={mode}, amp={args.amp}"
    )
    if args.dry_run == 1:
        return

    callback = make_live_plot_callback(params) if plot == 1 else None

    result = run_simulation(
        params,
        initial_condition=make_initial_condition(params, mode, args.amp),
        progress_callback=callback,
    )

    print(f"Finished: steps={result.t.size - 1}, t={result.t[-1]:.3f}")
    print(
        "Final metrics: "
        f"min={result.min_u[-1]:.6f}, "
        f"max={result.max_u[-1]:.6f}, "
        f"mean={result.mean_u[-1]:.6f}, "
        f"max_cfl={np.max(result.cfl):.6e}, "
        f"last_residual={result.residuals[-1]:.6e}"
    )

    if plot == 0:
        plot_final_result(result, params)
    else:
        plt.show()


if __name__ == "__main__":
    main()
