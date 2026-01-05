from dataclasses import dataclass, field
import logging
logger = logging.getLogger(__name__)
import torch.nn as nn
from trl import ModelConfig


@dataclass
class DebugModelConfig(ModelConfig):
    train_last_n_layers: int = field(
        default=0,
        metadata={"help": "Freeze all params, unfreeze last N transformer blocks (0 disables)."},
    )
    train_lm_head: bool = field(
        default=True,
        metadata={"help": "Whether to keep lm_head trainable when freezing."},
    )


def freeze_model_layers(model, train_last_n_layers: int = 0, train_lm_head: bool = True):
    # Freeze layers
    def _set_requires_grad(module, flag: bool):
        for p in module.parameters(recurse=True):
            p.requires_grad = flag
    
    def _get_qwen2vl_layers(model) -> nn.ModuleList:
        # Qwen2VL 常见放法（按优先级尝试）
        candidates = [
            ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
            # ("language_model.model.layers", getattr(getattr(getattr(model, "language_model", None), "model", None), "layers", None)),
            # ("language_model.layers", getattr(getattr(model, "language_model", None), "layers", None)),
        ]
        for name, layers in candidates:
            if isinstance(layers, nn.ModuleList):
                logger.warning(f"Using Qwen2VL layers from: {name} (len={len(layers)})")
                return layers
        raise RuntimeError("Cannot locate Qwen2VL transformer layers. Please inspect model structure and update _get_qwen2vl_layers().")
    
    n = train_last_n_layers
    if n and n > 0:
        _set_requires_grad(model, False)
    
        layers = _get_qwen2vl_layers(model)
        for layer in list(layers)[-n:]:
            _set_requires_grad(layer, True)
    
        if train_lm_head and hasattr(model, "lm_head") and model.lm_head is not None:
            _set_requires_grad(model.lm_head, True)
    
        def _numel(p):
            return getattr(p, "ds_numel", p.numel())  # ZeRO-3 下 ds_numel 更接近全局量
        
        trainable = sum(_numel(p) for p in model.parameters() if p.requires_grad)
        total = sum(_numel(p) for p in model.parameters())
        trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
        
        logger.warning(
            f"Froze model; trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
            f"({100*trainable/total:.2f}%), trainable_tensors={trainable_tensors}"
        )
