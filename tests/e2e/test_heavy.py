# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Heavy E2E regression: run DNABERT2 on 3 tasks, compare to baseline.

Regression (included in ``pytest tests/``):
  pytest tests/e2e/test_heavy.py::test_heavy_sanity_regression

Refresh pinned baseline (run explicitly by name):
  pytest tests/e2e/test_heavy.py::test_heavy_update_baseline -s
"""

from __future__ import annotations

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

DEFAULT_ATOL = 0.02


def _baseline_path(model_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    return (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / f"{safe}_sanity_baseline.csv"
    )


@pytest.fixture(scope="module")
def heavy_data_root(tmp_path_factory) -> Path:
    """Shared data directory; task init downloads datasets as needed."""
    return tmp_path_factory.mktemp("heavy_data")


@pytest.fixture
def heavy_config(heavy_data_root, tmp_path) -> BenchmarkConfig:
    return BenchmarkConfig(
        root_data_dir_path=str(heavy_data_root),
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


def _run_heavy_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    run_benchmark(config)
    return load_results(Path(config.csv_path))


def _write_baseline(results_df: pd.DataFrame, baseline_path: Path) -> None:
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


def test_heavy_sanity_regression(heavy_config):
    """Run DNABERT2 on 3 tasks and compare scores to the pinned baseline CSV."""
    model_name = heavy_config.model_name
    baseline_path = _baseline_path(model_name)

    if not baseline_path.is_file():
        pytest.fail(
            f"Missing baseline {baseline_path}. "
            "Create it with: "
            "pytest tests/e2e/test_heavy.py::test_heavy_update_baseline -s"
        )

    results_df = _run_heavy_benchmark(heavy_config)
    baseline_df = load_baseline(baseline_path)
    failures, warnings = compare_to_baseline(
        results_df,
        baseline_df,
        default_atol=DEFAULT_ATOL,
    )

    if warnings:
        print("\n".join(warnings))

    assert not failures, "Metric drift vs baseline:\n" + format_failures(failures)


def test_heavy_update_baseline(heavy_config):
    """Run DNABERT2 on 3 tasks and write scores to the pinned baseline CSV."""
    baseline_path = _baseline_path(heavy_config.model_name)
    results_df = _run_heavy_benchmark(heavy_config)
    _write_baseline(results_df, baseline_path)
    print(f"Wrote updated baseline to {baseline_path}")
