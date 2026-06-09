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

"""Compare benchmark CSV output against pinned baseline scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class MetricDiff:
    task: str
    metric: str
    expected: float
    actual: float
    abs_diff: float
    atol: float


def load_baseline(path: Path) -> pd.DataFrame:
    """Load baseline CSV with columns: task, metric, expected [, atol]."""
    df = pd.read_csv(path)
    required = {"task", "metric", "expected"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Baseline {path} missing columns: {sorted(missing)}")
    if "atol" not in df.columns:
        df["atol"] = float("nan")
    return df


def _infer_results_column(df: pd.DataFrame) -> str:
    value_cols = [c for c in df.columns if c not in {"task", "metric"}]
    if len(value_cols) != 1:
        raise ValueError(f"Expected one score column in results CSV, found {value_cols}")
    return value_cols[0]


def load_results(path: Path) -> pd.DataFrame:
    """Load a BenchmarkReport CSV; return task, metric, actual."""
    df = pd.read_csv(path)
    value_col = _infer_results_column(df)
    out = df[["task", "metric", value_col]].rename(columns={value_col: "actual"})
    return out


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in {"", "NO_RESULTS", "nan", "NaN"}:
        return True
    return pd.isna(value)


def compare_to_baseline(
    results: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    default_atol: float = 0.02,
) -> tuple[list[MetricDiff], list[str]]:
    """
    Compare actual results to baseline.

    Returns (failures, warnings). Rows with missing expected and actual are skipped.
    """
    merged = baseline.merge(results, on=["task", "metric"], how="left")
    failures: list[MetricDiff] = []
    warnings: list[str] = []

    for _, row in merged.iterrows():
        expected = row["expected"]
        actual = row.get("actual")
        atol = row["atol"] if pd.notna(row["atol"]) else default_atol

        if _is_missing(expected):
            continue
        if _is_missing(actual):
            failures.append(
                MetricDiff(
                    task=row["task"],
                    metric=row["metric"],
                    expected=float(expected),
                    actual=float("nan"),
                    abs_diff=float("inf"),
                    atol=float(atol),
                )
            )
            continue

        expected_f = float(expected)
        actual_f = float(actual)
        abs_diff = abs(actual_f - expected_f)
        if abs_diff > atol:
            failures.append(
                MetricDiff(
                    task=row["task"],
                    metric=row["metric"],
                    expected=expected_f,
                    actual=actual_f,
                    abs_diff=abs_diff,
                    atol=float(atol),
                )
            )

    baseline_keys = set(zip(baseline["task"], baseline["metric"]))
    result_keys = set(zip(results["task"], results["metric"]))
    extra = result_keys - baseline_keys
    if extra:
        warnings.append(
            f"{len(extra)} result row(s) not covered by baseline (first 5): "
            f"{sorted(extra)[:5]}"
        )

    return failures, warnings


def format_failures(failures: Iterable[MetricDiff]) -> str:
    lines = []
    for diff in failures:
        lines.append(
            f"  {diff.task} / {diff.metric}: expected {diff.expected:.6f}, "
            f"got {diff.actual:.6f} (|diff|={diff.abs_diff:.6f}, atol={diff.atol})"
        )
    return "\n".join(lines)
