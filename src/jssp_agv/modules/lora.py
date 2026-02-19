"""
LoRA (Low-Rank Adaptation) implementation for fine-tuning neural networks.

This module provides LoRA layers that can be applied to existing Linear layers
in neural networks, particularly for Graph Neural Networks used in JSSP.

References:
    LoRA: Low-Rank Adaptation of Large Language Models
    https://arxiv.org/abs/2106.09685


The LoRA module is not part of the papers results.
These are experimental features for future work on parameter-efficient fine-tuning of GNNs in JSSPT.
"""

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps an existing Linear layer.

    Adds low-rank matrices A and B such that the adapted weight is:
    W' = W + B @ A, where A is (r x in_features) and B is (out_features x r)

    Parameters
    ----------
    original_layer : nn.Linear
        The original linear layer to adapt
    rank : int
        Rank of the low-rank decomposition (typically 1-8)
    alpha : float
        Scaling factor for LoRA updates (default: 1.0)
    dropout : float
        Dropout probability for LoRA pathway (default: 0.0)
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through original layer + LoRA adaptation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Original layer output
        result = self.original_layer(x)

        # LoRA pathway: x @ A^T @ B^T * scaling
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

        return result + lora_out

    def merge_weights(self):
        """
        Merge LoRA weights into the original layer permanently.
        This makes the layer no longer trainable via LoRA.
        """
        if self.original_layer.weight.requires_grad:
            raise ValueError("Original layer is not frozen, cannot merge weights")

        # Compute LoRA delta: B @ A
        delta_W = (self.lora_B @ self.lora_A) * self.scaling

        # Update original weights
        self.original_layer.weight.data += delta_W

        # Zero out LoRA parameters
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adaptation.

    This creates a new Linear layer with LoRA from scratch, rather than
    wrapping an existing layer.

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    bias : bool
        If True, adds a learnable bias (default: True)
    rank : int
        Rank of the low-rank decomposition (default: 4)
    alpha : float
        Scaling factor for LoRA updates (default: 1.0)
    dropout : float
        Dropout probability for LoRA pathway (default: 0.0)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Create original linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Freeze original parameters
        for param in self.linear.parameters():
            param.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear layer + LoRA adaptation."""
        result = self.linear(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_out


class LoRAGNNLayer(nn.Module):
    """
    LoRA adaptation for Graph Neural Network layers.

    Applies LoRA to the MLP within GNN layers (e.g., GINConv).
    This wraps the entire MLP sequential module.

    Parameters
    ----------
    original_mlp : nn.Sequential
        The original MLP from a GNN layer (e.g., from GINConv)
    rank : int
        Rank for LoRA layers (default: 4)
    alpha : float
        Scaling factor for LoRA updates (default: 1.0)
    dropout : float
        Dropout probability for LoRA pathway (default: 0.0)
    """

    def __init__(
        self,
        original_mlp: nn.Sequential,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original_mlp = original_mlp
        self.rank = rank
        self.alpha = alpha
        self.dropout_p = dropout

        # Apply LoRA to all Linear layers in the MLP
        self.lora_layers = nn.ModuleDict()

        for i, module in enumerate(original_mlp):
            if isinstance(module, nn.Linear):
                # Freeze original layer
                for param in module.parameters():
                    param.requires_grad = False

                # Create LoRA adaptation
                lora_layer = LoRALayer(
                    original_layer=module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                self.lora_layers[str(i)] = lora_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through original MLP with LoRA adaptations.

        Args:
            x: Input tensor

        Returns:
            Output tensor after passing through MLP + LoRA
        """
        for i, module in enumerate(self.original_mlp):
            if str(i) in self.lora_layers:
                # Use LoRA-adapted layer
                x = self.lora_layers[str(i)](x)
            else:
                # Use original module (e.g., ReLU, BatchNorm)
                x = module(x)

        return x


def apply_lora_to_linear_layers(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """
    Apply LoRA to specific Linear layers in a model.

    Parameters
    ----------
    model : nn.Module
        The model to apply LoRA to
    rank : int
        Rank for LoRA layers (default: 4)
    alpha : float
        Scaling factor for LoRA updates (default: 1.0)
    dropout : float
        Dropout probability for LoRA pathway (default: 0.0)
    target_modules : Optional[List[str]]
        List of module names to apply LoRA to. If None, applies to all Linear layers.
        Example: ['policy_head', 'value_head']

    Returns
    -------
    nn.Module
        The model with LoRA layers applied
    """

    def should_apply_lora(name: str) -> bool:
        if target_modules is None:
            return True
        return any(target in name for target in target_modules)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            # Get parent module and attribute name
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Replace with LoRA layer
            lora_layer = LoRALayer(
                original_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            setattr(parent, attr_name, lora_layer)

    return model


def apply_lora_to_gnn(
    gnn_encoder: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA to GNN encoder (e.g., GINEncoder).

    Specifically targets the MLPs within GNN convolutional layers.

    Parameters
    ----------
    gnn_encoder : nn.Module
        The GNN encoder (e.g., GINEncoder)
    rank : int
        Rank for LoRA layers (default: 4)
    alpha : float
        Scaling factor for LoRA updates (default: 1.0)
    dropout : float
        Dropout probability for LoRA pathway (default: 0.0)

    Returns
    -------
    nn.Module
        The GNN encoder with LoRA layers applied
    """

    # Check if this is a GINEncoder with convs attribute
    if hasattr(gnn_encoder, "convs"):
        for i, conv_layer in enumerate(gnn_encoder.convs):
            # GINConv has an 'nn' attribute containing the MLP
            if hasattr(conv_layer, "nn") and isinstance(conv_layer.nn, nn.Sequential):
                # Apply LoRA to the MLP
                original_mlp = conv_layer.nn

                # Replace MLP with LoRA-adapted version
                lora_mlp = LoRAGNNLayer(
                    original_mlp=original_mlp,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                conv_layer.nn = lora_mlp

    return gnn_encoder


def freeze_non_lora_parameters(model: nn.Module) -> None:
    """
    Freeze all parameters except LoRA parameters.

    Parameters
    ----------
    model : nn.Module
        The model containing LoRA layers
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """
    Get all LoRA parameters from a model.

    Parameters
    ----------
    model : nn.Module or tuple
        The model containing LoRA layers, or a tuple of (separate_models, main_model)

    Returns
    -------
    List[nn.Parameter]
        List of LoRA parameters
    """
    lora_params = []

    # Handle tuple models (separate_models, main_model)
    if isinstance(model, tuple):
        separate_models, main_model = model

        # Process separate models
        if isinstance(separate_models, dict):
            for sub_model in separate_models.values():
                for name, param in sub_model.named_parameters():
                    if "lora_A" in name or "lora_B" in name:
                        lora_params.append(param)

        # Process main model
        for name, param in main_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)
    else:
        # Regular single model
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)

    return lora_params


def count_lora_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable LoRA parameters.

    Parameters
    ----------
    model : nn.Module or tuple
        The model containing LoRA layers, or a tuple of (separate_models, main_model)

    Returns
    -------
    int
        Number of trainable LoRA parameters
    """
    return sum(p.numel() for p in get_lora_parameters(model) if p.requires_grad)


def merge_all_lora_weights(model: nn.Module) -> None:
    """
    Merge all LoRA weights into their respective base layers.

    Parameters
    ----------
    model : nn.Module or tuple
        The model containing LoRA layers, or a tuple of (separate_models, main_model)
    """
    # Handle tuple models (separate_models, main_model)
    if isinstance(model, tuple):
        separate_models, main_model = model

        # Process separate models
        if isinstance(separate_models, dict):
            for sub_model in separate_models.values():
                for module in sub_model.modules():
                    if isinstance(module, (LoRALayer, LoRAGNNLayer)):
                        if hasattr(module, "merge_weights"):
                            module.merge_weights()

        # Process main model
        for module in main_model.modules():
            if isinstance(module, (LoRALayer, LoRAGNNLayer)):
                if hasattr(module, "merge_weights"):
                    module.merge_weights()
    else:
        # Regular single model
        for module in model.modules():
            if isinstance(module, (LoRALayer, LoRAGNNLayer)):
                if hasattr(module, "merge_weights"):
                    module.merge_weights()


def clean_lora_state_dict(state_dict: dict) -> dict:
    """
    Clean a state dict by removing LoRA artifacts and renaming merged keys.

    This function:
    1. Removes leftover "lora_layers" keys entirely
    2. Renames merged LoRA keys: "xxx.original_layer.weight" -> "xxx.weight"
    3. Keeps normal parameters unchanged

    Parameters
    ----------
    state_dict : dict
        The state_dict to clean (can be a checkpoint dict with "state_dict" key
        or a raw state_dict)

    Returns
    -------
    dict
        Cleaned state_dict without LoRA artifacts

    Example

    Use as dependency injection for ModelCheckpointLogger to save clean checkpoints without LoRA artifacts:
        --> ModelCheckpointLogger(self, save_dir: str, logger: TrainingLogger, keep_all_checkpoints: bool = False, *****clean_dict: THIS*****) -> None:
    """
    from collections import OrderedDict

    # Handle both checkpoint dicts and raw state dicts
    sd = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    new_sd = OrderedDict()

    for key, value in sd.items():
        # 1) REMOVE LoRA leftover keys entirely
        if "lora_layers" in key:
            continue

        # 2) RENAME merged LoRA keys: "xxx.original_layer.weight" -> "xxx.weight"
        if "original_layer" in key:
            new_key = key.replace(".original_layer", "")
            new_sd[new_key] = value
            continue

        # 3) Keep normal parameters unchanged
        new_sd[key] = value

    return new_sd


def merge_lora_weights_in_state_dict(
    state_dict: dict,
    rank: int | None = None,
    alpha: float = 1.0,
) -> dict:
    """
    Merge LoRA weights (lora_A and lora_B) into base weights in a state_dict.

    This function identifies LoRA parameter pairs in the state_dict and merges them
    into their corresponding base layer weights, producing a state_dict without
    LoRA parameters.

    Parameters
    ----------
    state_dict : dict
        The state_dict containing base weights and LoRA parameters
    rank : Optional[int]
        The rank used for LoRA. If None, will be inferred from lora_A shape
    alpha : float
        Scaling factor for LoRA updates (default: 1.0)

    Returns
    -------
    dict
        New state_dict with LoRA weights merged into base weights

    Example
    -------
    >>> # Load a checkpoint with LoRA
    >>> state_dict = torch.load("model_with_lora.pt")
    >>> # Merge LoRA weights into base weights
    >>> merged_state_dict = merge_lora_weights_in_state_dict(state_dict, alpha=1.0)
    >>> # Now load into a model without LoRA layers
    >>> model.load_state_dict(merged_state_dict)
    """
    merged_state_dict = {}
    processed_keys = set()

    # Find all lora_A keys to identify LoRA pairs
    lora_pairs = {}
    for key in state_dict.keys():
        if "lora_A" in key:
            # Extract base key by removing ".lora_A"
            base_key = key.replace(".lora_A", "")
            lora_b_key = key.replace("lora_A", "lora_B")

            if lora_b_key in state_dict:
                lora_pairs[base_key] = {"lora_A": key, "lora_B": lora_b_key}

    # Process all keys in state_dict
    for key, value in state_dict.items():
        # Skip LoRA parameters - they'll be merged
        if "lora_A" in key or "lora_B" in key:
            processed_keys.add(key)
            continue

        # Check if this is a base weight that has corresponding LoRA weights
        weight_key = key
        has_lora = False

        # Try to find matching LoRA pair
        for base_key, lora_keys in lora_pairs.items():
            # Check if this weight corresponds to the base key
            # Handle cases like "module.layer.weight" vs "module.layer"
            if key.startswith(base_key) and (
                key == f"{base_key}.weight"
                or key == f"{base_key}.original_layer.weight"
            ):
                has_lora = True
                lora_A = state_dict[lora_keys["lora_A"]]
                lora_B = state_dict[lora_keys["lora_B"]]

                # Infer rank if not provided
                if rank is None:
                    current_rank = lora_A.shape[0]
                else:
                    current_rank = rank

                # Compute scaling
                scaling = alpha / current_rank

                # Compute delta: B @ A * scaling
                delta_W = (lora_B @ lora_A) * scaling

                # Merge into base weight
                merged_weight = value + delta_W
                merged_state_dict[key] = merged_weight

                processed_keys.add(lora_keys["lora_A"])
                processed_keys.add(lora_keys["lora_B"])
                break

        if not has_lora:
            # No LoRA for this parameter, keep as is
            merged_state_dict[key] = value

        processed_keys.add(key)

    # Verify all keys were processed
    unprocessed = set(state_dict.keys()) - processed_keys
    if unprocessed:
        print(
            f"Warning: {len(unprocessed)} keys were not processed: {list(unprocessed)[:5]}..."
        )

    return merged_state_dict


def has_lora_weights(state_dict: dict) -> bool:
    """
    Check if a state_dict contains LoRA weights.

    Parameters
    ----------
    state_dict : dict
        The state_dict to check

    Returns
    -------
    bool
        True if LoRA weights are found, False otherwise
    """
    return any("lora_A" in key or "lora_B" in key for key in state_dict.keys())


def get_lora_info_from_state_dict(state_dict: dict) -> dict:
    """
    Extract information about LoRA layers from a state_dict.

    Parameters
    ----------
    state_dict : dict
        The state_dict to analyze

    Returns
    -------
    dict
        Dictionary with keys:
        - 'has_lora': bool
        - 'num_lora_layers': int
        - 'lora_params': int (total LoRA parameters)
        - 'base_params': int (total base parameters)
        - 'lora_layer_names': list of str
    """
    lora_layer_names = []
    lora_params = 0
    base_params = 0

    for key, value in state_dict.items():
        param_count = value.numel()

        if "lora_A" in key or "lora_B" in key:
            lora_params += param_count
            if "lora_A" in key:
                lora_layer_names.append(key)
        else:
            base_params += param_count

    return {
        "has_lora": len(lora_layer_names) > 0,
        "num_lora_layers": len(lora_layer_names),
        "lora_params": lora_params,
        "base_params": base_params,
        "lora_layer_names": lora_layer_names,
        "total_params": lora_params + base_params,
    }
