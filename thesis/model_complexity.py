from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelProfile:
    parameters: int
    trainable_parameters: int
    macs: int

    @property
    def gmac(self) -> float:
        return self.macs / 1_000_000_000


def profile_model(model: nn.Module, input_shape: tuple[int, int, int]) -> ModelProfile:
    """Count parameters and Conv2d/Linear multiply-accumulate operations."""
    hooks = []
    totals = {"macs": 0}
    was_training = model.training

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        output_tensor = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(module, nn.Conv2d):
            totals["macs"] += _conv2d_macs(module, output_tensor)
        elif isinstance(module, nn.Linear):
            totals["macs"] += _linear_macs(module, output_tensor)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))

    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros((1, *input_shape)))
    finally:
        for handle in hooks:
            handle.remove()
        model.train(was_training)

    return ModelProfile(
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        macs=totals["macs"],
    )


def build_complexity_row(
    model_name: str,
    profile: ModelProfile,
    input_shape: tuple[int, int, int],
    performance: dict[str, float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_name": model_name,
        "input_shape": "x".join(str(value) for value in input_shape),
        "parameters": profile.parameters,
        "trainable_parameters": profile.trainable_parameters,
        "macs": profile.macs,
        "gmac": profile.gmac,
    }
    if performance:
        for metric_name, value in performance.items():
            row[metric_name] = value
            if metric_name.endswith("_balanced_accuracy") and profile.gmac > 0:
                row[f"{metric_name}_per_gmac"] = value / profile.gmac
    return row


def _conv2d_macs(module: nn.Conv2d, output: torch.Tensor) -> int:
    batch_size, out_channels, out_h, out_w = output.shape
    kernel_h, kernel_w = module.kernel_size
    in_channels_per_group = module.in_channels // module.groups
    return (
        batch_size
        * out_channels
        * out_h
        * out_w
        * in_channels_per_group
        * kernel_h
        * kernel_w
    )


def _linear_macs(module: nn.Linear, output: torch.Tensor) -> int:
    batch_size = output.shape[0] if output.ndim > 1 else 1
    return batch_size * module.in_features * module.out_features
