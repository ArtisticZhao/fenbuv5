from __future__ import annotations

from cli import build_run_config, parse_cli_args
from config import write_template
from workflow import run_workflow


def main() -> None:
    args = parse_cli_args()
    try:
        if args.write_template is not None:
            write_template(args.write_template)
            print(f"Wrote configuration template: {args.write_template}")
            return

        config = build_run_config(args)
        run_workflow(config)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
