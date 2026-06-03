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

import argparse
import logging

from gfmbench_api.utils import logutils
from usage_examples.benchmark_runner import BenchmarkConfig, run_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GFM benchmark with configurable model and tasks"
    )
    parser.add_argument(
        "--root-data-dir-path",
        type=str,
        required=True,
        help="Root directory containing benchmark datasets",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to save benchmark results CSV",
    )
    parser.add_argument(
        "--report-algo-name",
        type=str,
        default="my_algo",
        help="Algorithm name for the benchmark report",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)",
    )
    parser.add_argument(
        "--mlm-head-path",
        type=str,
        default=None,
        help="Path to MLM head checkpoint (optional, for DNABERT2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DNABERT2",
        choices=["DNABERT2", "DNABERT", "Evo2"],
        help="Model to benchmark (default: DNABERT2)",
    )
    parser.add_argument(
        "--linear-prob",
        action="store_true",
        help="Use linear probing instead of full fine-tuning",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--disable-safe-model-call",
        action="store_true",
        help="Disable safe model call (use raw model forward pass)",
    )
    return parser.parse_args()


def main():
    logutils.init_logger()
    args = parse_args()

    sanity_check_mode = False
    if sanity_check_mode:
        logging.info("Running in sanity check mode (100 samples per dataset)")

    config = BenchmarkConfig(
        root_data_dir_path=args.root_data_dir_path,
        csv_path=args.csv_path,
        report_algo_name=args.report_algo_name,
        model_name=args.model,
        checkpoint_path=args.checkpoint_path,
        mlm_head_path=args.mlm_head_path,
        linear_probe=args.linear_prob,
        epochs=args.epochs,
        disable_safe_model_call=args.disable_safe_model_call,
        max_num_samples=100 if sanity_check_mode else 256,
    )
    run_benchmark(config)


if __name__ == "__main__":
    main()
