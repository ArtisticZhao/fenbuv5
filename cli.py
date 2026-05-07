from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from config import RunConfig, load_run_config, validate_run_config


@dataclass(frozen=True)
class CliArgs:
    config: Path | None
    write_template: Path | None
    simulate: bool | None
    hopf_indices: tuple[int, ...] | None
    output_dir: Path | None
    show_plots: bool | None
    save_plots: bool | None
    live_plots: bool | None


def parse_cli_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Find Hopf branch points and optionally run delayed reaction-diffusion simulations."
    )
    parser.add_argument("--config", type=Path, help="JSON configuration file.")
    parser.add_argument("--write-template", type=Path, help="Write a default JSON configuration template and exit.")
    parser.add_argument("--simulate", type=parse_bool, help="Override simulation.enabled.")
    parser.add_argument(
        "--hopf-indices",
        type=parse_indices,
        help="Comma-separated sorted Hopf candidate indices to simulate, e.g. 0,2.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override output.dir.")
    parser.add_argument("--show-plots", type=parse_bool, help="Override plots.show.")
    parser.add_argument("--save-plots", type=parse_bool, help="Override plots.save.")
    parser.add_argument(
        "--live-plots",
        type=parse_bool,
        help="Override plots.live. Enabling live plots also enables plots.show unless --show-plots false is set.",
    )
    args = parser.parse_args()
    return CliArgs(
        config=args.config,
        write_template=args.write_template,
        simulate=args.simulate,
        hopf_indices=args.hopf_indices,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
        save_plots=args.save_plots,
        live_plots=args.live_plots,
    )


def build_run_config(args: CliArgs) -> RunConfig:
    config = load_run_config(args.config) if args.config is not None else RunConfig()

    if args.simulate is not None:
        config = replace(config, simulation=replace(config.simulation, enabled=args.simulate))
    if args.hopf_indices is not None:
        config = replace(config, simulation=replace(config.simulation, hopf_indices=args.hopf_indices))
    if args.output_dir is not None:
        config = replace(config, output=replace(config.output, dir=args.output_dir))
    if args.show_plots is not None:
        config = replace(config, plots=replace(config.plots, show=args.show_plots))
    if args.save_plots is not None:
        config = replace(config, plots=replace(config.plots, save=args.save_plots))
    if args.live_plots is not None:
        show = config.plots.show
        if args.live_plots and args.show_plots is None:
            show = True
        config = replace(config, plots=replace(config.plots, live=args.live_plots, show=show))

    validate_run_config(config)
    return config


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def parse_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hopf indices must be comma-separated integers") from exc
    if not indices:
        raise argparse.ArgumentTypeError("hopf indices must not be empty")
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("hopf indices must be non-negative")
    return indices
