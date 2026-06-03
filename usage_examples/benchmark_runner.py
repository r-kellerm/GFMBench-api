# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared benchmark execution logic for run_benchmark.py and tests.

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from gfmbench_api.benchmark_report import BenchmarkReport
from gfmbench_api.tasks.concrete.bend_vep_disease_task import BendVEPDisease
from gfmbench_api.tasks.concrete.bend_vep_expression_task import BendVEPExpression
from gfmbench_api.tasks.concrete.brca1_task import BRCA1Task
from gfmbench_api.tasks.concrete.clinvar_indel_task import IndelClinvarTask
from gfmbench_api.tasks.concrete.clinvar_vepeval_task import VepevalClinvarTask
from gfmbench_api.tasks.concrete.gue_promoter_all_task import GuePromoterAllTask
from gfmbench_api.tasks.concrete.gue_splice_site_task import GueSpliceSiteTask
from gfmbench_api.tasks.concrete.gue_tf_all_task import GueTranscriptionFactorTask
from gfmbench_api.tasks.concrete.loleve_causal_eqtl_task import LoleveCausalEqtlTask
from gfmbench_api.tasks.concrete.lrb_causal_eqtl_task import LRBCausalEqtlTask
from gfmbench_api.tasks.concrete.lrb_pathogenic_omim_task import (
    LrbVariantEffectPathogenicOmimTask,
)
from gfmbench_api.tasks.concrete.songlab_clinvar_task import SonglabClinvarTask
from gfmbench_api.tasks.concrete.traitgym_complex_task import TraitGymComplexTask
from gfmbench_api.tasks.concrete.traitgym_mendelian_task import TraitGymMendelianTask
from gfmbench_api.tasks.concrete.variant_benchmarks_coding_task import (
    VariantBenchmarksCodingTask,
)
from gfmbench_api.tasks.concrete.variant_benchmarks_common_vs_rare_task import (
    VariantBenchmarksCommonVsRareTask,
)
from gfmbench_api.tasks.concrete.variant_benchmarks_expression_task import (
    VariantBenchmarksExpressionTask,
)
from gfmbench_api.tasks.concrete.variant_benchmarks_meqtl_task import (
    VariantBenchmarksMEQTLTask,
)
from gfmbench_api.tasks.concrete.variant_benchmarks_non_coding_task import (
    VariantBenchmarksNonCodingTask,
)
from gfmbench_api.tasks.concrete.variant_benchmarks_sqtl_task import (
    VariantBenchmarksSQTLTask,
)
from usage_examples.trainers import GFMFinetuner

TaskFactory = Callable[..., Any]

TASK_REGISTRY: dict[str, TaskFactory] = {
    "vepeval_clinvar": VepevalClinvarTask,
    "clinvar_indel": IndelClinvarTask,
    "loleve_causal_eqtl": LoleveCausalEqtlTask,
    "brca1": BRCA1Task,
    "gue_transcription_factor": GueTranscriptionFactorTask,
    "gue_promoter_all": GuePromoterAllTask,
    "gue_splice_site": GueSpliceSiteTask,
    "bend_variant_effects_expression": BendVEPExpression,
    "bend_variant_effects_disease": BendVEPDisease,
    "songlab_clinvar": SonglabClinvarTask,
    "var_bench_coding_pathogenicity": VariantBenchmarksCodingTask,
    "var_bench_non_coding_pathogenicity": VariantBenchmarksNonCodingTask,
    "var_bench_expression": VariantBenchmarksExpressionTask,
    "var_bench_common_vs_rare": VariantBenchmarksCommonVsRareTask,
    "var_bench_meqtl": VariantBenchmarksMEQTLTask,
    "var_bench_sqtl": VariantBenchmarksSQTLTask,
    "traitgym_complex": TraitGymComplexTask,
    "traitgym_mendelian": TraitGymMendelianTask,
    "lrb_variant_effect_pathogenic_omim": LrbVariantEffectPathogenicOmimTask,
    "lrb_variant_effect_causal_eqtl": LRBCausalEqtlTask,
}

ALL_TASK_NAMES: list[str] = [
    "vepeval_clinvar",
    "clinvar_indel",
    "loleve_causal_eqtl",
    "brca1",
    "gue_transcription_factor",
    "gue_promoter_all",
    "gue_splice_site",
    "bend_variant_effects_expression",
    "bend_variant_effects_disease",
    "songlab_clinvar",
    "var_bench_coding_pathogenicity",
    "var_bench_non_coding_pathogenicity",
    "var_bench_expression",
    "var_bench_common_vs_rare",
    "var_bench_meqtl",
    "var_bench_sqtl",
    "traitgym_complex",
    "traitgym_mendelian",
    "lrb_variant_effect_pathogenic_omim",
    "lrb_variant_effect_causal_eqtl",
]

DEFAULT_HEAVY_TASKS: list[str] = [
    "gue_promoter_all",
    "songlab_clinvar",
    "var_bench_coding_pathogenicity",
]

DEFAULT_HEAVY_MODEL = "DNABERT2"


@dataclass
class BenchmarkConfig:
    root_data_dir_path: str
    csv_path: str
    report_algo_name: str = "my_algo"
    model_name: str = DEFAULT_HEAVY_MODEL
    checkpoint_path: Optional[str] = None
    mlm_head_path: Optional[str] = None
    linear_probe: bool = False
    epochs: int = 3
    max_num_samples: Optional[int] = 256
    task_batch_size: int = 16
    training_batch_size: int = 8
    disable_safe_model_call: bool = False
    num_workers: int = 0
    seed: int = 0
    task_names: Optional[list[str]] = None


def get_model_registry() -> dict[str, dict[str, Any]]:
    """Return model registry; Evo2 is omitted when optional deps are missing."""
    from usage_examples.sanity_models.dna_bert2_model import DNABERT2Model
    from usage_examples.sanity_models.dna_bert_model import DNABERTModel

    registry: dict[str, dict[str, Any]] = {
        "DNABERT2": {"class": DNABERT2Model, "max_length": 2500},
        "DNABERT": {"class": DNABERTModel, "max_length": 500},
    }
    try:
        from usage_examples.sanity_models.evo2_model import Evo2BioNeMoModel

        registry["Evo2"] = {"class": Evo2BioNeMoModel, "max_length": 8192}
    except ImportError:
        logging.debug("Evo2 unavailable (optional Megatron/BioNeMo deps missing)")
    return registry


def load_mlm_head_only(model, mlm_head_path: str) -> None:
    """Load only MLM head weights from a checkpoint (DNABERT2)."""
    logging.info("Loading MLM head only from: %s", mlm_head_path)
    state = torch.load(mlm_head_path, map_location=model.device)

    if "mlm_head_state_dict" in state:
        model.mlm_head.load_state_dict(state["mlm_head_state_dict"])
        model.mlm_head_loaded = True
        logging.info("Loaded MLM head weights from mlm_head_state_dict")
    elif "mlm_model_state_dict" in state:
        mlm_state = state["mlm_model_state_dict"]
        mlm_head_state = {
            key[4:]: value
            for key, value in mlm_state.items()
            if key.startswith("cls.")
        }
        if mlm_head_state:
            model.mlm_head.load_state_dict(mlm_head_state, strict=False)
            model.mlm_head_loaded = True
            logging.info("Loaded MLM head weights from legacy mlm_model_state_dict")
        else:
            logging.warning("No MLM head weights found in legacy checkpoint")
    else:
        logging.error("No MLM head weights found in %s", mlm_head_path)
        logging.error("  Available keys: %s", list(state.keys()))
        raise ValueError("No MLM head weights found in MLM head checkpoint")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _instantiate_model(
    registry: dict[str, dict[str, Any]],
    model_name: str,
    device: str,
) -> tuple[Any, int, type]:
    if model_name not in registry:
        raise ValueError(
            f"Unknown model {model_name!r}. Available: {sorted(registry.keys())}"
        )
    entry = registry[model_name]
    model_class = entry["class"]
    max_length = entry["max_length"]
    model = model_class(device=device, max_length=max_length)
    return model, max_length, model_class


def _resolve_task_names(task_names: Optional[list[str]]) -> list[str]:
    names = list(ALL_TASK_NAMES if task_names is None else task_names)
    unknown = [name for name in names if name not in TASK_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown task(s): {unknown}. Known: {sorted(TASK_REGISTRY)}"
        )
    return names


def run_benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    """Run configured tasks and write scores to ``config.csv_path``."""
    _set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s", device)

    registry = get_model_registry()
    if config.model_name not in registry:
        raise ValueError(
            f"Unknown model: {config.model_name}. "
            f"Supported models: {sorted(registry.keys())}"
        )
    if config.model_name == "Evo2" and not config.linear_probe:
        raise ValueError(
            "Evo2 does not support full fine-tuning. Use linear_probe=True."
        )

    _, max_length, _ = _instantiate_model(registry, config.model_name, device)
    task_names = _resolve_task_names(config.task_names)

    task_config = {
        "max_sequence_length": max_length,
        "batch_size": config.task_batch_size,
        "max_num_samples": config.max_num_samples,
        "disable_safe_model_call": config.disable_safe_model_call,
        "num_workers": config.num_workers,
    }
    if config.max_num_samples is not None:
        logging.info("Limiting to %s samples per dataset", config.max_num_samples)

    tasks = [
        TASK_REGISTRY[name](
            root_data_dir_path=config.root_data_dir_path,
            task_config=task_config,
        )
        for name in task_names
    ]

    training_params = {
        "num_epochs": config.epochs,
        "lr": 3e-5,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "only_proj_layer": config.linear_probe,
        "batch_size": config.training_batch_size,
    }

    report = BenchmarkReport(csv_path=config.csv_path)
    logging.info("Model: %s", config.model_name)
    logging.info("Tasks (%d): %s", len(tasks), task_names)
    logging.info(
        "Training mode: %s (%d epochs)",
        "Linear Probing" if config.linear_probe else "Full Fine-tuning",
        config.epochs,
    )

    for task in tasks:
        task_name = task.get_task_name()
        logging.info("=" * 50)
        logging.info("Running task: %s", task_name)
        logging.info("=" * 50)

        model, _, _ = _instantiate_model(registry, config.model_name, device)
        if config.checkpoint_path:
            logging.info("Loading checkpoint: %s", config.checkpoint_path)
            model.load_checkpoint(config.checkpoint_path)
        if config.mlm_head_path and hasattr(model, "mlm_head"):
            logging.info("Loading MLM head from: %s", config.mlm_head_path)
            load_mlm_head_only(model, config.mlm_head_path)

        task_attrs = task.get_task_attributes()
        if task_attrs.get("has_finetuning_data", False):
            generator = torch.Generator()
            generator.manual_seed(config.seed)
            train_loader = DataLoader(
                task.get_finetune_dataset(),
                batch_size=training_params["batch_size"],
                shuffle=True,
                generator=generator,
                num_workers=config.num_workers,
            )
            finetuner = GFMFinetuner(
                model=model,
                train_loader=train_loader,
                hidden_dim=model.get_hidden_dim(),
                num_labels=task_attrs["num_labels"],
                num_epochs=training_params["num_epochs"],
                lr=training_params["lr"],
                optimizer_name=training_params["optimizer"],
                weight_decay=training_params["weight_decay"],
                only_proj_layer=training_params["only_proj_layer"],
                is_variant_effect_prediction=task_attrs.get(
                    "is_variant_effect_prediction", False
                ),
                device=device,
            )
            logging.info("Fine-tuning...")
            test_model = finetuner.fine_tune()
        else:
            logging.info("Zero-shot task detected. Skipping fine-tuning.")
            test_model = model

        logging.info("Evaluating...")
        test_model.eval()
        scores = task.eval_test_set(test_model)
        logging.info("Test scores for %s: %s", task_name, scores)

        report.add_scores(task_name, config.report_algo_name, scores)
        report.save_csv()
        logging.info("Saved progress to: %s", config.csv_path)

    logging.info("=" * 50)
    logging.info("Benchmark complete. Report saved to: %s", config.csv_path)
    logging.info("=" * 50)
    logging.info("%s", report)
    return report
