"""
E2E download tests: verify tasks fetch data into an empty data directory.

Simulates a fresh user with no local benchmark cache. Requires network access.

Enable with:
  RUN_DOWNLOAD_TESTS=1 pytest tests/e2e/test_download.py -m download
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Type

import pytest

from gfmbench_api.tasks.concrete.gue_promoter_all_task import GuePromoterAllTask
from gfmbench_api.tasks.concrete.variant_benchmarks_coding_task import (
    VariantBenchmarksCodingTask,
)

pytestmark = pytest.mark.download

DOWNLOAD_TASK_CONFIG = {
    "max_num_samples": 4,
    "num_workers": 0,
    "batch_size": 2,
    "max_sequence_length": 128,
}


def _download_env_ready() -> tuple[bool, str]:
    if os.environ.get("RUN_DOWNLOAD_TESTS") != "1":
        return False, "Set RUN_DOWNLOAD_TESTS=1 to enable download tests"
    return True, ""


@pytest.fixture
def download_data_root(tmp_path) -> Path:
    ready, reason = _download_env_ready()
    if not ready:
        pytest.skip(reason)
    root = tmp_path / "data"
    root.mkdir()
    assert not any(root.iterdir()), "expected empty data directory"
    return root


def _assert_required_files(task_dir: Path, required_files: Sequence[str]) -> None:
    missing = [name for name in required_files if not (task_dir / name).is_file()]
    assert not missing, f"Missing downloaded files in {task_dir}: {missing}"


@pytest.mark.parametrize(
    ("task_cls", "task_dir_name", "required_files"),
    [
        pytest.param(
            GuePromoterAllTask,
            "gue_promoter_all",
            ("train.csv", "dev.csv", "test.csv"),
            id="gue_promoter_all",
        ),
        pytest.param(
            VariantBenchmarksCodingTask,
            "var_bench_coding_pathogenicity",
            ("dataset_dict.json",),
            id="var_bench_coding_pathogenicity",
        ),
    ],
)
def test_task_downloads_from_empty_data_dir(
    download_data_root: Path,
    task_cls: Type,
    task_dir_name: str,
    required_files: Sequence[str],
):
    """Task init on an empty root should download HF data and expose non-empty splits."""
    task = task_cls(
        root_data_dir_path=str(download_data_root),
        task_config=dict(DOWNLOAD_TASK_CONFIG),
    )

    task_dir = download_data_root / task_dir_name
    assert task_dir.is_dir(), f"Expected task data directory {task_dir}"
    _assert_required_files(task_dir, required_files)

    assert task.get_task_name() == task_dir_name
    assert len(task.test_dataset) > 0

    attrs = task.get_task_attributes()
    if attrs.get("has_finetuning_data"):
        finetune = task.get_finetune_dataset()
        assert finetune is not None and len(finetune) > 0


def test_gue_download_is_idempotent(download_data_root: Path):
    """Second init should reuse cached files without error."""
    kwargs = {
        "root_data_dir_path": str(download_data_root),
        "task_config": dict(DOWNLOAD_TASK_CONFIG),
    }
    task1 = GuePromoterAllTask(**kwargs)
    task_dir = download_data_root / "gue_promoter_all"
    _assert_required_files(task_dir, ("train.csv", "dev.csv", "test.csv"))

    csv_mtimes = {
        path.name: path.stat().st_mtime_ns
        for path in task_dir.glob("*.csv")
    }

    task2 = GuePromoterAllTask(**kwargs)
    _assert_required_files(task_dir, ("train.csv", "dev.csv", "test.csv"))
    for name, mtime_ns in csv_mtimes.items():
        assert (task_dir / name).stat().st_mtime_ns == mtime_ns

    assert len(task2.test_dataset) == len(task1.test_dataset)
