"""Train/infer orchestration using core + optional Oracle."""

from __future__ import annotations

import random
from pathlib import Path

from core.domain import SchedulingDataset
from core.evaluation import (
    evaluate_all_benchmark_datasets,
    render_html_report,
    update_benchmark_markdown,
)
from core.rl.trainer import infer_conversions, train_with_imitation_then_ppo


def _load_settings():
    import json

    return json.loads(Path("config/settings.json").read_text(encoding="utf-8"))


def _datasets_from_benchmark(benchmark_dir: str | Path) -> SchedulingDataset:
    return SchedulingDataset.from_csv_dir(benchmark_dir)


def _datasets_from_db(from_key: str, to_key: str) -> list[SchedulingDataset]:
    from biz.oracle_repo import fetch_timekeys_in_range, load_dataset_from_db

    keys = fetch_timekeys_in_range(from_key, to_key)
    if not keys:
        keys = [from_key]
    return [load_dataset_from_db(k) for k in keys]


def run_training(
    from_rule_timekey: str | None = None,
    to_rule_timekey: str | None = None,
    rule_timekey: str | None = None,
    benchmark_dataset: str | None = None,
    steps: int = 50_000,
    run_test_eval: bool = True,
) -> Path:
    settings = _load_settings()
    imitation_epochs = settings.get("training", {}).get("imitation_epochs", 20)
    model_dir = settings.get("artifacts", {}).get("model_dir", "artifacts/models")

    if benchmark_dataset:
        ds_list = [_datasets_from_benchmark(benchmark_dataset)]
    else:
        fk = from_rule_timekey or rule_timekey
        tk = to_rule_timekey or rule_timekey or fk
        if not fk:
            raise ValueError("Specify --from-timekey/--to-timekey, --timekey, or --benchmark-dataset")
        if fk == tk:
            try:
                ds_list = _datasets_from_db(fk, tk)
            except Exception:
                ds_list = [_datasets_from_benchmark(f"benchmarks/benchmark_01")]
        else:
            try:
                ds_list = _datasets_from_db(fk, tk)
            except Exception:
                bench_root = Path("benchmarks")
                ds_list = [SchedulingDataset.from_csv_dir(p) for p in sorted(bench_root.glob("benchmark_*"))[:3]]

    model_path = train_with_imitation_then_ppo(
        ds_list,
        total_timesteps=steps,
        imitation_epochs=imitation_epochs,
        model_dir=model_dir,
    )

    if run_test_eval:
        _run_benchmark_eval(model_dir)

    return model_path


def _run_benchmark_eval(model_dir: str):
    model_path = Path(model_dir) / "ppo_scheduling"

    def loader(bench_dir: Path):
        ds = SchedulingDataset.from_csv_dir(bench_dir)
        return infer_conversions(ds, model_path)

    results = evaluate_all_benchmark_datasets("benchmarks", policy_loader=loader)
    update_benchmark_markdown(results)
    settings = _load_settings()
    report_dir = settings.get("artifacts", {}).get("report_dir", "artifacts/reports")
    render_html_report(results, Path(report_dir) / "benchmark_report.html")


def run_inference(
    rule_timekey: str | None = None,
    benchmark_dataset: str | None = None,
    output: str | None = None,
) -> Path:
    settings = _load_settings()
    model_dir = Path(settings.get("artifacts", {}).get("model_dir", "artifacts/models"))
    model_path = model_dir / "ppo_scheduling"

    if benchmark_dataset:
        dataset = _datasets_from_benchmark(benchmark_dataset)
        rtk = dataset.rule_timekey
    else:
        rtk = rule_timekey
        if not rtk:
            try:
                from biz.oracle_repo import fetch_max_timekey

                rtk = fetch_max_timekey()
            except Exception:
                rtk = "2026051707000000"
        try:
            from biz.oracle_repo import load_dataset_from_db

            dataset = load_dataset_from_db(rtk)
        except Exception:
            dataset = SchedulingDataset.from_csv_dir("benchmarks/benchmark_01")
            rtk = dataset.rule_timekey

    conversions = infer_conversions(dataset, model_path)
    out_path = Path(output or settings.get("artifacts", {}).get("inference_dir", "artifacts/inference") + "/allocation.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = dataset.to_conversions_df(conversions)
    df.to_csv(out_path, index=False)

    try:
        from biz.oracle_repo import save_conversions

        save_conversions(conversions, rtk)
    except Exception:
        pass

    return out_path
