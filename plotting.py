from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from config import PlotConfig
from hopf_branch import (
    HopfBranchData,
    HopfPolynomialFit,
    HopfSearchConfig,
    plot_hopf_branches,
)
from reaction_diffusion_delay import SimulationParams, SimulationResult


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


def render_hopf_plots(
    branches: list[HopfBranchData],
    config: HopfSearchConfig,
    plot_config: PlotConfig,
    output_dir: Path,
    *,
    polynomial_fits: list[HopfPolynomialFit],
) -> list[Path]:
    if not plot_config.hopf:
        return []

    figs = plot_hopf_branches(
        branches,
        config,
        show=False,
        polynomial_fits=polynomial_fits,
    )
    saved_paths: list[Path] = []
    if plot_config.save:
        names = ("hopf_root_branches.png", "hopf_s_values.png", "hopf_candidates.png")
        output_dir.mkdir(parents=True, exist_ok=True)
        for fig, name in zip(figs, names):
            path = output_dir / name
            fig.savefig(path, dpi=200, bbox_inches="tight")
            saved_paths.append(path)

    if plot_config.show and "agg" not in plt.get_backend().lower():
        plt.show(block=False)
        plt.pause(0.001)
    else:
        for fig in figs:
            plt.close(fig)
    return saved_paths


def plot_final_result(
    result: SimulationResult,
    params: SimulationParams,
    title: str,
    plot_config: PlotConfig,
    *,
    save_path: Path | None = None,
) -> Path | None:
    if not plot_config.final:
        return None

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
    saved_path = None
    if plot_config.save and save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        saved_path = save_path

    if plot_config.show and "agg" not in plt.get_backend().lower():
        plt.show(block=False)
        plt.pause(0.001)
    else:
        plt.close(fig)
    return saved_path


def show_open_plots(plot_config: PlotConfig) -> None:
    if plot_config.show and "agg" not in plt.get_backend().lower():
        plt.show()
