from __future__ import annotations

import torch.nn as nn

from models.pneumonia_net import PneumoniaNet


_MODEL_NAMES = (
    "pneumonia_net",
    "resnet18",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "mobilenet_v3_large",
)


def available_models() -> tuple[str, ...]:
    return _MODEL_NAMES


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "pneumonia_net":
        return PneumoniaNet()

    models = _torchvision_models()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        return model
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return model
    if name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return model
    raise ValueError(f"Unknown model '{name}'. Available models: {', '.join(_MODEL_NAMES)}")


def expected_channels(model_name: str) -> int:
    return 1 if model_name == "pneumonia_net" else 3


def configure_trainable_layers(model: nn.Module, model_name: str, mode: str) -> nn.Module:
    model_name = model_name.lower()
    mode = mode.lower()
    if mode not in {"all", "head", "last_block"}:
        raise ValueError("Trainable mode must be one of: all, head, last_block")

    for parameter in model.parameters():
        parameter.requires_grad = mode == "all"
    if mode == "all" or model_name == "pneumonia_net":
        return model

    if model_name in {"resnet18", "resnet50"}:
        if mode == "last_block":
            _unfreeze_module(model.layer4)
        _unfreeze_module(model.fc)
    elif model_name == "densenet121":
        if mode == "last_block":
            _unfreeze_module(model.features.denseblock4)
            _unfreeze_module(model.features.norm5)
        _unfreeze_module(model.classifier)
    elif model_name in {"efficientnet_b0", "mobilenet_v3_large"}:
        if mode == "last_block":
            _unfreeze_module(model.features[-1])
        _unfreeze_module(model.classifier)
    else:
        raise ValueError(f"Cannot configure unknown model '{model_name}'")
    return model


def freeze_backbone(model: nn.Module, model_name: str) -> nn.Module:
    return configure_trainable_layers(model, model_name, "head")


def _unfreeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def _torchvision_models():
    try:
        from torchvision import models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for comparison models. Install it before training "
            "resnet18, resnet50, densenet121, efficientnet_b0, or mobilenet_v3_large."
        ) from exc
    return models
