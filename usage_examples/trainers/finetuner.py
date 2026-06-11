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

# This module does not embed third-party data download URLs.
import contextlib
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from gfmbench_api.utils.caching_utils import SequenceInferenceCache
from usage_examples.trainers.model_wrapper import GFMWithProjection


def _is_cuda_device(device) -> bool:
    """
    Accept str ("cuda", "cuda:0"), torch.device, or bare int (treated as a CUDA
    ordinal, matching torch.cuda.set_device semantics).
    """
    if isinstance(device, int):
        return True
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


def _amp_available(device) -> bool:
    """
    AMP is usable when the device is CUDA, CUDA is actually present, and
    torch.amp exposes autocast. Returns False on CPU/MPS.
    """
    if not _is_cuda_device(device):
        return False
    if not torch.cuda.is_available():
        return False
    return hasattr(torch, "amp") and hasattr(torch.amp, "autocast")


def _pick_amp_dtype() -> torch.dtype:
    """
    Prefer bfloat16 on hardware that supports it (Ampere+ / Hopper / etc.):
    bf16 keeps fp32 dynamic range and needs no loss scaling. Fall back to
    fp16 + GradScaler on older GPUs.
    """
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


class GFMFinetuner:
    """
    Fine-tuning module for genomic foundation models.
    Handles training of projection layers and full models for classification tasks.
    """
    
    def __init__(
        self, 
        model, 
        train_loader,
        hidden_dim,
        num_labels,
        num_epochs,
        lr,
        optimizer_name="AdamW",
        weight_decay=0.01,
        only_proj_layer=True,
        is_variant_effect_prediction=False,
        disable_cache=False,
        device='cpu',
        use_amp=None,
        amp_dtype=None,
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            model: BaseGFMModel instance
            train_loader: DataLoader for training data
            hidden_dim: hidden dimension size of the model
            num_labels: number of classification labels
            num_epochs: number of training epochs
            lr: learning rate
            optimizer_name: optimizer name ('Adam', 'AdamW', 'SGD')
            weight_decay: weight decay for regularization
            only_proj_layer: if True, only train projection layer; if False, train full model
            is_variant_effect_prediction: if True, task uses variant/ref sequence pairs
            disable_cache: if True, skip frozen backbone forward cache during linear probing
            device: torch device
            use_amp: enable CUDA mixed-precision for the full fine-tune path. None ->
                auto: on iff CUDA is available and torch.amp is installed. Pass False
                to force off. Has no effect on the linear-probe path, which stays fp32
                so the cached backbone output remains numpy-serializable.
            amp_dtype: autocast dtype. None -> auto: bfloat16 when supported (no loss
                scaling needed), else float16 (with GradScaler).
        """
        self.model = model
        self.train_loader = train_loader
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.only_proj_layer = only_proj_layer
        self.is_variant_effect_prediction = is_variant_effect_prediction
        self.disable_cache = disable_cache
        self.device = device

        if use_amp is None:
            use_amp = _amp_available(device)
        elif use_amp and not _amp_available(device):
            print("Warning: use_amp=True but CUDA/torch.amp unavailable; disabling AMP.")
            use_amp = False
        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype if amp_dtype is not None else _pick_amp_dtype()

        if self.use_amp:
            if self.only_proj_layer:
                print(
                    "AMP requested but not applied on the cached linear-probe path "
                    "(backbone is cached in fp32 for numpy-safe storage)."
                )
            else:
                print(f"AMP enabled for full fine-tuning (dtype={self.amp_dtype}).")

        # Create projection layer for classification
        # For variant effect tasks, input is concatenated embeddings (hidden_dim * 2)
        proj_input_dim = self.hidden_dim * 2 if is_variant_effect_prediction else self.hidden_dim
        self.projection = nn.Linear(proj_input_dim, self.num_labels).to(device)
    
    def _get_optimizer(self, params):
        """
        Create optimizer based on configuration.
        
        Args:
            params: model parameters to optimize
            
        Returns:
            torch.optim.Optimizer: configured optimizer
        """
        optimizer_name = self.optimizer_name.lower()
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from: 'adam', 'adamw', 'sgd'")
            
    
    def fine_tune(self):
        """
        Fine-tune the model or only the projection layer for classification tasks.
        
        Returns:
            GFMWithProjection: wrapped model with trained projection layer
        """
        # Determine which parameters to optimize
        if self.only_proj_layer:
            params = self.projection.parameters()
        else:
            params = list(self.model.parameters()) + list(self.projection.parameters())
        
        # Initialize optimizer
        optimizer = self._get_optimizer(params)
        
        # Loss function for classification
        criterion = torch.nn.CrossEntropyLoss()

        # AMP is applied only to the full fine-tune path. On the linear-probe
        # path the backbone output is cached as CPU numpy, and numpy has no
        # bfloat16 dtype, so autocasting the cached forward would crash on store.
        # The cache already eliminates the backbone cost across epochs there, so
        # keeping that path in fp32 costs effectively nothing.
        amp_active = self.use_amp and not self.only_proj_layer
        needs_scaler = amp_active and self.amp_dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler) if amp_active else None

        def _autocast():
            if not amp_active:
                return contextlib.nullcontext()
            return torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)

        # Set to training mode
        if not self.only_proj_layer:
            self.model.train()
        else:
            self.model.eval()  # Keep model in eval mode if only training projection
        self.projection.train()

        fwd_cache = SequenceInferenceCache() if self.only_proj_layer else None

        # When only the projection layer is trained, freeze backbone params for
        # the duration of training. Inputs (token ids) never require grad, so with
        # the backbone frozen its forward yields requires_grad=False outputs
        # naturally -- no autograd graph is built and the per-batch no_grad()/
        # detach() dance is unnecessary. Cache hits return fresh tensors rebuilt
        # from numpy (also grad-free). State is restored in the finally block so
        # the caller's model is not permanently mutated.
        original_requires_grad: Optional[List[Tuple[torch.nn.Parameter, bool]]] = None
        if self.only_proj_layer:
            original_requires_grad = [
                (p, p.requires_grad) for p in self.model.parameters()
            ]
            for p, _ in original_requires_grad:
                p.requires_grad_(False)

        avg_loss = float("nan")
        try:
            # Training loop
            for epoch in range(self.num_epochs):
                total_loss = 0.0
                num_batches = 0

                progress_bar = tqdm(self.train_loader, desc=f"Fine-tuning epoch {epoch+1}/{self.num_epochs}")
                for batch in progress_bar:
                    with _autocast():
                        if self.is_variant_effect_prediction:
                            # Variant effect task: (variant_seqs, ref_seqs, labels, conditional_input)
                            variant_sequences, ref_sequences, labels, conditional_input = batch
                            labels = labels.to(self.device)

                            if self.only_proj_layer:
                                var_repr = fwd_cache.cached_call(
                                    self.model._sequence_to_representative,
                                    variant_sequences,
                                    disable=self.disable_cache,
                                )
                                ref_repr = fwd_cache.cached_call(
                                    self.model._sequence_to_representative,
                                    ref_sequences,
                                    disable=self.disable_cache,
                                )
                            else:
                                var_repr = self.model._sequence_to_representative(variant_sequences)
                                ref_repr = self.model._sequence_to_representative(ref_sequences)

                            # Concatenate variant and reference representations
                            sequence_repr = torch.cat([var_repr, ref_repr], dim=1)
                        else:
                            # Single sequence task: (sequences, labels, conditional_input)
                            sequences, labels, conditional_input = batch
                            labels = labels.to(self.device)

                            if self.only_proj_layer:
                                sequence_repr = fwd_cache.cached_call(
                                    self.model._sequence_to_representative,
                                    sequences,
                                    disable=self.disable_cache,
                                )
                            else:
                                sequence_repr = self.model._sequence_to_representative(sequences)

                        logits = self.projection(sequence_repr)
                        loss = criterion(logits, labels)

                    optimizer.zero_grad(set_to_none=True)
                    if needs_scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    # Update loss tracking
                    total_loss += loss.item()
                    num_batches += 1
                    avg_loss = total_loss / num_batches

                    # Update progress bar with average loss
                    progress_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        finally:
            if original_requires_grad is not None:
                for p, orig in original_requires_grad:
                    p.requires_grad_(orig)

            if fwd_cache is not None:
                fwd_cache.clear()

        if self.num_epochs > 0:
            print(f"Fine-tuning completed. Final average loss: {avg_loss:.4f}")
        else:
            print("Fine-tuning skipped")
        
        # Return wrapped model with projection layer
        wrapped_model = GFMWithProjection(
            self.model, self.projection, disable_cache=self.disable_cache
        )
        wrapped_model.eval()  # Set to eval mode after training
        return wrapped_model


