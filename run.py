"""CLI entrypoint.

Examples:
  python run.py train --from-timekey 20251020070000 --to-timekey 20251020120000 --steps 50000
  python run.py infer --timekey 20251020070000
  python run.py infer                    # RULE_TIMEKEY = MAX from DB
  python run.py train --benchmark-dataset benchmarks/benchmark_01 --steps 50000
  python run.py infer --benchmark-dataset benchmarks/benchmark_01 \
                      --output artifacts/inference/allocation.csv
  python run.py eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from biz.pipeline import load_settings, run_eval, run_infer, run_train

DEFAULT_SETTINGS = "config/settings.json"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--settings", default=DEFAULT_SETTINGS, help="Path to settings.json")
    p.add_argument(
        "--mode",
        choices=("plan-only", "wip-static", "dynamic", "all"),
        help="Scheduling model. plan-only: ignore WIP. wip-static: single "
             "snapshot with WIP cap (default). dynamic: multi-period WIP flow + "
             "switch cost. 'all' is only valid for eval.",
    )


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--from-timekey", dest="from_timekey")
    p.add_argument("--to-timekey", dest="to_timekey")
    p.add_argument("--timekey", dest="rule_timekey")
    p.add_argument("--benchmark-dataset", dest="benchmark_dataset")
    p.add_argument("--steps", type=int)


def _add_infer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--timekey", dest="rule_timekey")
    p.add_argument("--benchmark-dataset", dest="benchmark_dataset")
    p.add_argument("--output", dest="output_csv")
    p.add_argument(
        "--dump-snapshot",
        action="store_true",
        help="Save infer input as JSON (artifacts/inference/snapshots) and reload it",
    )
    p.add_argument(
        "--snapshot-path",
        dest="snapshot_path",
        help="Override JSON snapshot file path (implies structured infer log)",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="run.py", description="Equipment switching scheduler")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Imitation + PPO training")
    _add_common(p_train)
    _add_train_args(p_train)

    p_infer = sub.add_parser("infer", help="Run inference for a snapshot")
    _add_common(p_infer)
    _add_infer_args(p_infer)

    p_eval = sub.add_parser("eval", help="Benchmark evaluation (no DB)")
    _add_common(p_eval)

    args = parser.parse_args(argv)
    if not Path(args.settings).exists():
        print(f"settings file not found: {args.settings}", file=sys.stderr)
        return 2
    settings = load_settings(args.settings)

    if args.command == "train":
        if args.mode == "all":
            parser.error("--mode all is only valid for eval")
        result = run_train(
            settings,
            from_timekey=args.from_timekey,
            to_timekey=args.to_timekey,
            rule_timekey=args.rule_timekey,
            benchmark_dataset=args.benchmark_dataset,
            steps=args.steps,
            mode=args.mode,
        )
    elif args.command == "infer":
        if args.mode == "all":
            parser.error("--mode all is only valid for eval")
        result = run_infer(
            settings,
            rule_timekey=args.rule_timekey,
            benchmark_dataset=args.benchmark_dataset,
            output_csv=args.output_csv,
            mode=args.mode,
            dump_snapshot=args.dump_snapshot or bool(args.snapshot_path),
            snapshot_path=args.snapshot_path,
        )
    elif args.command == "eval":
        result = run_eval(settings, mode=args.mode)
    else:  # pragma: no cover
        parser.error(f"unknown command: {args.command}")
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
