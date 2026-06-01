"""Benchmark evaluation and HTML report generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import Template

from core.domain import ConversionRecord, SchedulingDataset
from core.optimizer import HeuristicSolver, ImprovedGreedySolver, OptimalSolver, load_ground_truth_conversions
from core.simulator import SchedulingSimulator


@dataclass
class EvalMetrics:
    name: str
    avg_achievement_rate: float
    conversion_count: int
    achievement_by_oper: dict[str, float]
    achievement_by_model: dict[str, float]


def _format_oper_key(key: tuple[str, str]) -> str:
    return f"{key[0]}|{key[1]}"


def evaluate_conversions(dataset: SchedulingDataset, conversions: list[ConversionRecord], name: str) -> EvalMetrics:
    sim = SchedulingSimulator(dataset)
    result = sim.simulate(conversions)
    return EvalMetrics(
        name=name,
        avg_achievement_rate=result.avg_achievement_rate,
        conversion_count=result.conversion_count,
        achievement_by_oper={_format_oper_key(k): v for k, v in result.achievement_by_oper.items()},
        achievement_by_model=result.achievement_by_model,
    )


def evaluate_dataset(
    dataset_dir: str | Path,
    policy_conversions: list[ConversionRecord] | None = None,
    policy_name: str = "RL",
) -> dict[str, EvalMetrics]:
    base = Path(dataset_dir)
    dataset = SchedulingDataset.from_csv_dir(base)
    errors = dataset.validate()
    if errors:
        raise ValueError("; ".join(errors))

    gt_path = base / "ground_truth.json"
    optimal_convs = load_ground_truth_conversions(gt_path) if gt_path.exists() else OptimalSolver().solve(dataset)
    heuristic_convs = HeuristicSolver().solve(dataset)
    rl_convs = policy_conversions if policy_conversions is not None else ImprovedGreedySolver().solve(dataset)

    return {
        "optimal": evaluate_conversions(dataset, optimal_convs, "optimal"),
        "heuristic": evaluate_conversions(dataset, heuristic_convs, "heuristic"),
        policy_name.lower(): evaluate_conversions(dataset, rl_convs, policy_name),
    }


def evaluate_all_benchmark_datasets(
    benchmarks_root: str | Path = "benchmarks",
    policy_loader=None,
) -> dict[str, dict[str, EvalMetrics]]:
    root = Path(benchmarks_root)
    results: dict[str, dict[str, EvalMetrics]] = {}
    for bench_dir in sorted(root.glob("benchmark_*")):
        if not bench_dir.is_dir():
            continue
        name = bench_dir.name
        policy_convs = None
        if policy_loader:
            policy_convs = policy_loader(bench_dir)
        results[name] = evaluate_dataset(bench_dir, policy_conversions=policy_convs)
    return results


def update_benchmark_markdown(
    results: dict[str, dict[str, EvalMetrics]],
    path: str | Path = "MODEL_BENCHMARK.md",
) -> None:
    lines = [
        "# Model Benchmark Record",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Benchmark | Optimal Avg | Heuristic Avg | RL Avg | Optimal Conv | RL Conv |",
        "|-----------|-------------|---------------|--------|--------------|---------|",
    ]
    for bench, metrics in results.items():
        opt = metrics.get("optimal")
        heu = metrics.get("heuristic")
        rl = metrics.get("rl") or metrics.get("policy")
        if not opt:
            continue
        rl_m = rl or heu
        lines.append(
            f"| {bench} | {opt.avg_achievement_rate:.4f} | "
            f"{heu.avg_achievement_rate if heu else 0:.4f} | "
            f"{rl_m.avg_achievement_rate if rl_m else 0:.4f} | "
            f"{opt.conversion_count} | {rl_m.conversion_count if rl_m else 0} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <title>Scheduling Benchmark Report</title>
  <style>
    body { font-family: sans-serif; margin: 2rem; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
    th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
    th { background: #f0f0f0; }
    h2 { margin-top: 2rem; }
  </style>
</head>
<body>
  <h1>Equipment Transition Scheduling — Benchmark Report</h1>
  <p>Generated: {{ generated_at }}</p>
  {% for bench, rows in benchmarks.items() %}
  <h2>{{ bench }}</h2>
  <table>
    <tr><th>Method</th><th>Avg Plan Achievement</th><th>Conversion Count</th></tr>
    {% for row in rows %}
    <tr>
      <td>{{ row.method }}</td>
      <td>{{ "%.2f"|format(row.achievement * 100) }}%</td>
      <td>{{ row.conversions }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endfor %}
</body>
</html>
"""


def render_html_report(
    results: dict[str, dict[str, EvalMetrics]],
    output_path: str | Path = "artifacts/reports/benchmark_report.html",
) -> Path:
    benchmarks: dict[str, list[dict]] = {}
    for bench, metrics in results.items():
        rows = []
        for method in ("optimal", "heuristic", "rl"):
            m = metrics.get(method)
            if m:
                rows.append(
                    {
                        "method": method,
                        "achievement": m.avg_achievement_rate,
                        "conversions": m.conversion_count,
                    }
                )
        benchmarks[bench] = rows

    html = Template(REPORT_TEMPLATE).render(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        benchmarks=benchmarks,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def results_to_json(results: dict[str, dict[str, EvalMetrics]]) -> str:
    payload = {}
    for bench, metrics in results.items():
        payload[bench] = {k: asdict(v) for k, v in metrics.items()}
    return json.dumps(payload, indent=2, ensure_ascii=False)
