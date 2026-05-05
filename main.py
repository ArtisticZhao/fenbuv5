from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

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
    return parser.parse_args()


def make_original_params() -> SimulationParams:
    return SimulationParams(
        d1=1.0,
        d2=3.0,
        r1=0.1,
        tau=2.0,
        epsilon=0.8,
        L=np.pi,
        Nx=101,
        dt=0.05,
        Tend=300.0,
        sweep_tol=1.0e-4,
        stagnation_tol=1.0e-5,
        max_sweeps=200,
    )


def original_initial_condition(x: np.ndarray) -> np.ndarray:
    return 1.0 + 0.01 * np.cos(2.0 * np.pi * x / np.pi)


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

    params = make_original_params()
    callback = make_live_plot_callback(params) if plot == 1 else None

    result = run_simulation(
        params,
        initial_condition=original_initial_condition,
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
