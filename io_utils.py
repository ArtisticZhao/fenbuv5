from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import RunConfig, config_to_json_dict
from hopf_branch import HopfPoint, HopfSearchConfig
from reaction_diffusion_delay import SimulationParams, SimulationResult


def save_result_data(
    result: SimulationResult,
    params: SimulationParams,
    mode: int,
    amplitude: float,
    output_path: Path,
    *,
    hopf_point: HopfPoint,
    hopf_config: HopfSearchConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        cfl_warning_threshold=params.cfl_warning_threshold,
        initial_mode=mode,
        amplitude=amplitude,
        hopf_enabled=True,
        hopf_epsilon=hopf_point.epsilon,
        hopf_omega=hopf_point.omega,
        hopf_n=hopf_point.n,
        hopf_k=hopf_point.k,
        hopf_mu=hopf_point.mu,
        hopf_s_value=hopf_point.s_value,
        hopf_fu=hopf_config.fu,
        hopf_domain_factor=hopf_config.domain_factor,
    )


def save_resolved_config(config: RunConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config_to_json_dict(config), handle, indent=2)
        handle.write("\n")
