from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import json
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import numpy as np

from hopf_branch import HopfSearchConfig
from reaction_diffusion_delay import SimulationParams


@dataclass(frozen=True)
class HopfRunConfig:
    n_list: tuple[int, ...] = HopfSearchConfig.n_list
    k_list: tuple[int, ...] = HopfSearchConfig.k_list
    eps_min: float = HopfSearchConfig.eps_min
    eps_max_margin: float = HopfSearchConfig.eps_max_margin
    omega_min: float = HopfSearchConfig.omega_min
    omega_max: float = HopfSearchConfig.omega_max
    num_eps: int = HopfSearchConfig.num_eps
    num_omega: int = HopfSearchConfig.num_omega
    min_hopf_omega: float = HopfSearchConfig.min_hopf_omega
    max_roots_per_eps: int = HopfSearchConfig.max_roots_per_eps
    max_total_branches: int = HopfSearchConfig.max_total_branches
    branch_gap_abs: float = HopfSearchConfig.branch_gap_abs
    branch_gap_rel: float = HopfSearchConfig.branch_gap_rel
    branch_max_fill: int = HopfSearchConfig.branch_max_fill
    s_tol: float = HopfSearchConfig.s_tol
    polynomial_degree: int = HopfSearchConfig.polynomial_degree
    polynomial_window: int = HopfSearchConfig.polynomial_window
    polynomial_plot_points: int = HopfSearchConfig.polynomial_plot_points
    duplicate_eps_tol: float = HopfSearchConfig.duplicate_eps_tol


@dataclass(frozen=True)
class SimulationRunConfig:
    enabled: bool = True
    hopf_indices: tuple[int, ...] | None = None
    amplitude: float = 0.01
    default_initial_mode: int = 2


@dataclass(frozen=True)
class PlotConfig:
    show: bool = False
    save: bool = True
    live: bool = False
    hopf: bool = True
    final: bool = True


@dataclass(frozen=True)
class OutputConfig:
    dir: Path = Path("output")
    save_data: bool = True
    save_resolved_config: bool = True


@dataclass(frozen=True)
class RunConfig:
    model: SimulationParams = field(default_factory=SimulationParams)
    hopf: HopfRunConfig = field(default_factory=HopfRunConfig)
    simulation: SimulationRunConfig = field(default_factory=SimulationRunConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


T = TypeVar("T")


def default_run_config() -> RunConfig:
    return RunConfig()


def load_run_config(path: Path) -> RunConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a JSON object")
    config = _dataclass_from_dict(RunConfig, raw)
    validate_run_config(config)
    return config


def write_template(path: Path) -> None:
    config = default_run_config()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config_to_json_dict(config), handle, indent=2)
        handle.write("\n")


def config_to_json_dict(config: RunConfig) -> dict[str, Any]:
    return _json_ready(asdict(config))


def validate_run_config(config: RunConfig) -> None:
    model = config.model
    hopf = config.hopf
    simulation = config.simulation
    plots = config.plots

    if model.L <= 0:
        raise ValueError("model.L must be positive")
    if model.Nx < 3:
        raise ValueError("model.Nx must be at least 3")
    if model.dt <= 0:
        raise ValueError("model.dt must be positive")
    if model.Tend < 0:
        raise ValueError("model.Tend must be non-negative")
    if model.tau <= 0:
        raise ValueError("model.tau must be positive")
    if not 0 < model.epsilon < model.tau:
        raise ValueError("model.epsilon must satisfy 0 < epsilon < tau")
    if model.sweep_tol <= 0:
        raise ValueError("model.sweep_tol must be positive")
    if model.stagnation_tol < 0:
        raise ValueError("model.stagnation_tol must be non-negative")
    if model.max_sweeps <= 0:
        raise ValueError("model.max_sweeps must be positive")
    if model.cfl_warning_threshold <= 0:
        raise ValueError("model.cfl_warning_threshold must be positive")

    if not hopf.n_list:
        raise ValueError("hopf.n_list must not be empty")
    if not hopf.k_list:
        raise ValueError("hopf.k_list must not be empty")
    if any(n <= 0 for n in hopf.n_list):
        raise ValueError("hopf.n_list values must be positive")
    if any(k < 0 for k in hopf.k_list):
        raise ValueError("hopf.k_list values must be non-negative")
    if hopf.eps_min <= 0:
        raise ValueError("hopf.eps_min must be positive")
    if hopf.eps_max_margin <= 0:
        raise ValueError("hopf.eps_max_margin must be positive")
    if hopf.eps_min >= model.tau - hopf.eps_max_margin:
        raise ValueError("hopf epsilon search range must satisfy eps_min < tau - eps_max_margin")
    if hopf.omega_min <= 0 or hopf.omega_max <= hopf.omega_min:
        raise ValueError("hopf omega range must satisfy 0 < omega_min < omega_max")
    if hopf.num_eps < 2:
        raise ValueError("hopf.num_eps must be at least 2")
    if hopf.num_omega < 2:
        raise ValueError("hopf.num_omega must be at least 2")
    if hopf.min_hopf_omega < 0:
        raise ValueError("hopf.min_hopf_omega must be non-negative")
    if hopf.max_roots_per_eps <= 0:
        raise ValueError("hopf.max_roots_per_eps must be positive")
    if hopf.max_total_branches <= 0:
        raise ValueError("hopf.max_total_branches must be positive")
    if hopf.branch_gap_abs <= 0 or hopf.branch_gap_rel < 0:
        raise ValueError("hopf branch gap values must be positive/non-negative")
    if hopf.branch_max_fill < 0:
        raise ValueError("hopf.branch_max_fill must be non-negative")
    if hopf.s_tol <= 0:
        raise ValueError("hopf.s_tol must be positive")
    if hopf.polynomial_degree < 1:
        raise ValueError("hopf.polynomial_degree must be at least 1")
    if hopf.polynomial_window < 2:
        raise ValueError("hopf.polynomial_window must be at least 2")
    if hopf.polynomial_plot_points < 2:
        raise ValueError("hopf.polynomial_plot_points must be at least 2")
    if hopf.duplicate_eps_tol < 0:
        raise ValueError("hopf.duplicate_eps_tol must be non-negative")

    if simulation.amplitude < 0:
        raise ValueError("simulation.amplitude must be non-negative")
    if simulation.default_initial_mode <= 0:
        raise ValueError("simulation.default_initial_mode must be positive")
    if simulation.hopf_indices is not None:
        if not simulation.hopf_indices:
            raise ValueError("simulation.hopf_indices must be null or a non-empty list")
        if any(index < 0 for index in simulation.hopf_indices):
            raise ValueError("simulation.hopf_indices values must be non-negative")

    if plots.live and not plots.show:
        raise ValueError("plots.live requires plots.show=true because live plots need an interactive window")

    if not np.isfinite(model.L / np.pi):
        raise ValueError("model.L / pi must be finite")


def _dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    allowed = {field.name: field for field in fields(cls)}
    type_hints = get_type_hints(cls)
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(f"unknown configuration field(s) for {cls.__name__}: {', '.join(unknown)}")

    values = {}
    for field in fields(cls):
        if field.name not in data:
            continue
        values[field.name] = _coerce_value(type_hints[field.name], data[field.name], field.name)
    return cls(**values)


def _coerce_value(annotation: Any, value: Any, name: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object")
        return _dataclass_from_dict(annotation, value)

    if annotation is Path:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string path")
        return Path(value)

    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a JSON list")
        item_type = args[0] if args else Any
        return tuple(_coerce_scalar(item_type, item, name) for item in value)

    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a JSON list")
        item_type = args[0] if args else Any
        return [_coerce_scalar(item_type, item, name) for item in value]

    if origin is not None and type(None) in args:
        if value is None:
            return None
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) != 1:
            return value
        return _coerce_value(non_none[0], value, name)

    return _coerce_scalar(annotation, value, name)


def _coerce_scalar(annotation: Any, value: Any, name: str) -> Any:
    if annotation is Any:
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        return value
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        return value
    if annotation is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        return value
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
