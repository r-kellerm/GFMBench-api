"""
Heavy E2E regression: run DNABERT2 on 3 tasks, compare to baseline.

Requires:
  RUN_HEAVY_TESTS=1
  GFMBENCH_DATA_ROOT=<path to benchmark datasets>

Regenerate baseline:
  RUN_HEAVY_TESTS=1 UPDATE_HEAVY_BASELINE=1 pytest tests/e2e/test_heavy.py -m heavy -s
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pytest

from tests.e2e.baseline_utils import (
    compare_to_baseline,
    format_failures,
    load_baseline,
    load_results,
)
from usage_examples.benchmark_runner import (
    BenchmarkConfig,
    DEFAULT_HEAVY_MODEL,
    DEFAULT_HEAVY_TASKS,
    run_benchmark,
)

pytestmark = pytest.mark.heavy

DEFAULT_ATOL = 0.02


def _baseline_path(model_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    return (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / f"{safe}_sanity_baseline.csv"
    )


def _heavy_env_ready() -> tuple[bool, str]:
    if os.environ.get("RUN_HEAVY_TESTS") != "1":
        return False, "Set RUN_HEAVY_TESTS=1 to enable heavy regression tests"
    data_root = os.environ.get("GFMBENCH_DATA_ROOT")
    if not data_root or not Path(data_root).is_dir():
        return False, "Set GFMBENCH_DATA_ROOT to an existing data directory"
    return True, ""


@pytest.fixture
def heavy_config(tmp_path) -> BenchmarkConfig:
    ready, reason = _heavy_env_ready()
    if not ready:
        pytest.skip(reason)

    return BenchmarkConfig(
        root_data_dir_path=os.environ["GFMBENCH_DATA_ROOT"],
        csv_path=str(tmp_path / "heavy_results.csv"),
        report_algo_name=f"{DEFAULT_HEAVY_MODEL.lower()}_heavy",
        model_name=DEFAULT_HEAVY_MODEL,
        linear_probe=True,
        epochs=1,
        max_num_samples=100,
        task_names=list(DEFAULT_HEAVY_TASKS),
        task_batch_size=8,
        training_batch_size=8,
        disable_safe_model_call=True,
    )


def test_heavy_sanity_regression(heavy_config):
    model_name = heavy_config.model_name
    baseline_path = _baseline_path(model_name)

    run_benchmark(heavy_config)
    results_df = load_results(Path(heavy_config.csv_path))

    if os.environ.get("UPDATE_HEAVY_BASELINE") == "1":
        baseline_rows = []
        for _, row in results_df.iterrows():
            if pd.isna(row["actual"]):
                continue
            baseline_rows.append(
                {
                    "task": row["task"],
                    "metric": row["metric"],
                    "expected": row["actual"],
                    "atol": DEFAULT_ATOL,
                }
            )
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
        pytest.skip(f"Wrote updated baseline to {baseline_path}")

    if not baseline_path.is_file():
        pytest.fail(
            f"Missing baseline {baseline_path}. "
            "Run once with UPDATE_HEAVY_BASELINE=1 to create it."
        )

    baseline_df = load_baseline(baseline_path)
    failures, warnings = compare_to_baseline(
        results_df,
        baseline_df,
        default_atol=DEFAULT_ATOL,
    )

    if warnings:
        print("\n".join(warnings))

    assert not failures, "Metric drift vs baseline:\n" + format_failures(failures)
