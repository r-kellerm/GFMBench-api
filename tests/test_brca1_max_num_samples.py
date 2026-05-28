# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``BRCA1Task`` honouring ``task_config['max_num_samples']``.

Previously ``BRCA1Task`` read its sample cap from ``task_config['max_samples']`` and
stored it on ``self.max_samples``. The rest of the framework (and ``BaseGFMTask``)
uses ``max_num_samples`` / ``self.max_num_samples``, so callers that followed the
documented contract silently got the full ~3,893-sample dataset on every run.
"""

from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd
import pytest


SEQ_LEN = 8
DF_ROWS = 50


@pytest.fixture
def patched_brca1():
    """Stub out disk I/O so ``BRCA1Task.__init__`` runs end-to-end in milliseconds.

    Yields a no-op context where:
      - ``os.path.exists`` always returns True (download branch is skipped)
      - ``pd.read_parquet`` returns a 50-row synthetic BRCA1-like dataframe
      - ``pyfaidx.Fasta`` returns a tiny in-memory chromosome stub
      - ``pad_sequence_centered_variant`` returns a fixed ``SEQ_LEN``-bp sequence
        whose centre base matches every row's ``ref`` allele (so the in-loop
        ref-vs-genome assertion passes for every iteration).
    """
    df = pd.DataFrame(
        {
            "chrom": ["chr17"] * DF_ROWS,
            "pos": list(range(1, DF_ROWS + 1)),
            "ref": ["A"] * DF_ROWS,
            "alt": ["T"] * DF_ROWS,
            "label": [0] * DF_ROWS,
        }
    )
    fake_seq = "A" * SEQ_LEN

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "gfmbench_api.tasks.concrete.brca1_task.os.path.exists",
                return_value=True,
            )
        )
        stack.enter_context(
            patch(
                "gfmbench_api.tasks.concrete.brca1_task.pd.read_parquet",
                return_value=df,
            )
        )
        stack.enter_context(patch("pyfaidx.Fasta", return_value={"chr17": fake_seq}))
        stack.enter_context(
            patch(
                "gfmbench_api.tasks.concrete.brca1_task.pad_sequence_centered_variant",
                return_value=fake_seq,
            )
        )
        yield df


def _build_task(task_config):
    """Import + construct inside the test so the patches are active during init."""
    from gfmbench_api.tasks.concrete.brca1_task import BRCA1Task

    return BRCA1Task(root_data_dir_path="/does/not/exist", task_config=task_config)


def test_max_num_samples_caps_test_dataset(patched_brca1):
    """``task_config['max_num_samples']=N`` -> exactly N samples in test_dataset."""
    cap = 7
    task = _build_task({"max_num_samples": cap, "max_sequence_length": SEQ_LEN})

    assert task.max_num_samples == cap
    assert len(task.test_dataset) == cap


def test_unset_max_num_samples_keeps_all_rows(patched_brca1):
    """No cap in task_config -> the full (mocked) dataset comes through."""
    task = _build_task({"max_sequence_length": SEQ_LEN})

    assert task.max_num_samples is None
    assert len(task.test_dataset) == DF_ROWS


def test_legacy_max_samples_key_has_no_effect(patched_brca1):
    """The pre-fix wrong key ``max_samples`` must NOT cap the dataset.

    Regression guard: if someone re-introduces ``cfg.get("max_samples", ...)``,
    this test will fail because the cap will start applying again.
    """
    task = _build_task({"max_samples": 3, "max_sequence_length": SEQ_LEN})

    assert task.max_num_samples is None
    assert len(task.test_dataset) == DF_ROWS
