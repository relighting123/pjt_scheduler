#!/usr/bin/env python3
"""CLI entry: train, infer, eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Equipment transition scheduling (IL + PPO)")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Imitation learning warm-start then PPO training")
    train_p.add_argument("--from-timekey", dest="from_rule_timekey", default=None)
    train_p.add_argument("--to-timekey", dest="to_rule_timekey", default=None)
    train_p.add_argument("--timekey", dest="rule_timekey", default=None)
    train_p.add_argument("--steps", type=int, default=50_000)
    train_p.add_argument("--benchmark-dataset", default=None)
    train_p.add_argument("--no-eval", action="store_true", help="Skip post-train benchmark evaluation")

    infer_p = sub.add_parser("infer", help="Run inference and write RTD_CONV allocation CSV")
    infer_p.add_argument("--timekey", dest="rule_timekey", default=None)
    infer_p.add_argument("--benchmark-dataset", default=None)
    infer_p.add_argument("--output", default=None)

    infer_all_p = sub.add_parser(
        "infer-all",
        help="Infer on all benchmarks, save CSVs, and open HTML summary report",
    )
    infer_all_p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-benchmark allocation CSV (default: artifacts/inference)",
    )
    infer_all_p.add_argument(
        "--report",
        default=None,
        help="HTML report path (default: artifacts/reports/inference_summary.html)",
    )

    sub.add_parser("eval", help="Evaluate all benchmarks (optimal vs heuristic vs RL)")

    args = parser.parse_args(argv)

    if args.command == "train":
        from biz.pipeline import run_training

        path = run_training(
            from_rule_timekey=args.from_rule_timekey,
            to_rule_timekey=args.to_rule_timekey,
            rule_timekey=args.rule_timekey,
            benchmark_dataset=args.benchmark_dataset,
            steps=args.steps,
            run_test_eval=not args.no_eval,
        )
        print(f"Model saved: {path}")
        return 0

    if args.command == "infer":
        from biz.pipeline import run_inference

        out = run_inference(
            rule_timekey=args.rule_timekey,
            benchmark_dataset=args.benchmark_dataset,
            output=args.output,
        )
        print(f"Allocation written: {out}")
        return 0

    if args.command == "infer-all":
        from biz.pipeline import run_inference_all_benchmarks

        report, csv_paths = run_inference_all_benchmarks(
            output_dir=args.output_dir,
            report_path=args.report,
        )
        print(f"HTML summary: {report.resolve()}")
        for name, path in csv_paths.items():
            print(f"  {name}: {path.resolve()}")
        return 0

    if args.command == "eval":
        from core.evaluation import evaluate_all_benchmark_datasets, render_html_report, update_benchmark_markdown
        from core.rl.trainer import infer_conversions
        from core.domain import SchedulingDataset

        model_path = Path("artifacts/models/ppo_scheduling")

        def loader(bench_dir: Path):
            ds = SchedulingDataset.from_csv_dir(bench_dir)
            if model_path.with_suffix(".zip").exists() or model_path.exists():
                return infer_conversions(ds, model_path)
            from core.optimizer import ImprovedGreedySolver

            return ImprovedGreedySolver().solve(ds)

        results = evaluate_all_benchmark_datasets("benchmarks", policy_loader=loader)
        update_benchmark_markdown(results)
        report = render_html_report(results)
        print(f"Benchmark markdown updated. HTML report: {report}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
