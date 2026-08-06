"""
Model Factory
=============

Registry-based model factory with ``@register_model`` decorator.

The Trainer never knows which backbone it is training — it only calls
:func:`create_model`. Adding a new backbone requires **one** decorated
function (≈10 lines) and zero changes to training code.

Usage::

    from src.models.factory import create_model, list_models

    model = create_model("resnet50", pretrained=True, num_classes=6)
    print(list_models())  # ['resnet50', 'densenet121', ...]

Adding a new backbone::

    @register_model("convnext_tiny")
    def _convnext_tiny(pretrained: bool, num_classes: int, **kw) -> nn.Module:
        import timm
        model = timm.create_model("convnext_tiny", pretrained=pretrained)
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        return model
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn


# ============================================================
# Registry
# ============================================================

_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model builder function.

    Args:
        name: String identifier used in YAML configs and ``create_model()``.

    Example::

        @register_model("my_model")
        def _my_model(pretrained, num_classes, **kw):
            ...
    """

    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        _MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def create_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: int = 6,
    **kwargs: Any,
) -> nn.Module:
    """
    Create a model by name using the registry.

    This is the **only** model creation API the Trainer uses.

    Args:
        model_name: Registered model identifier.
        pretrained: Whether to load pretrained ImageNet weights.
        num_classes: Number of output classes.
        **kwargs: Extra keyword arguments forwarded to the builder.

    Returns:
        An ``nn.Module`` with the correct output head.

    Raises:
        ValueError: If ``model_name`` is not registered.
    """
    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: [{available}]"
        )
    return _MODEL_REGISTRY[model_name](
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )


def list_models() -> List[str]:
    """Return sorted list of all registered model names."""
    return sorted(_MODEL_REGISTRY.keys())


# ============================================================
# Built-in Backbone Registrations
# ============================================================


@register_model("resnet50")
def _resnet50(
    pretrained: bool = True,
    num_classes: int = 6,
    **kwargs: Any,
) -> nn.Module:
    """ResNet-50 (torchvision) with replaced FC head."""
    import torchvision.models as tv_models

    weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@register_model("densenet121")
def _densenet121(
    pretrained: bool = True,
    num_classes: int = 6,
    **kwargs: Any,
) -> nn.Module:
    """DenseNet-121 (torchvision) with replaced classifier head."""
    import torchvision.models as tv_models

    weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = tv_models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


@register_model("efficientnet_b0")
def _efficientnet_b0(
    pretrained: bool = True,
    num_classes: int = 6,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> nn.Module:
    """EfficientNet-B0 (timm) with replaced classifier head."""
    import timm

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


@register_model("swin_tiny")
def _swin_tiny(
    pretrained: bool = True,
    num_classes: int = 6,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> nn.Module:
    """Swin Transformer Tiny (timm) with replaced head."""
    import timm

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model
