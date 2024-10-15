import torch
import torch.nn as nn
from scipy.stats import norm
from typing import *
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from functools import partial
from ..utils.sampling import SamplingDataset
from .propagate import propagate
from src.utils.helper import Config
import gc


class Pruner:
    def __init__(self, layers, ratio: float, method="unstructed") -> None:
        self.ratio = ratio
        self.method = method
        self.layers = layers
        self.pruning_mask = {}
        self.pruning_indices = {}

    def wanda(self, layer, inputs, outputs):
        current_weight = layer.weight.data
        X = inputs[0]  # (batch_size, seq_dim, input_dim)
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        nsamples = X.shape[0]  # (batch_size)
        if len(X.shape) == 3:
            X = X.reshape((-1, X.shape[-1]))  # (batch_size * seq_dim, input_dim)

        X = X.t()  # (input_dim, batch_size * seq_dim)
        scaler_row = torch.norm(X, p=2, dim=1) ** 2 / nsamples

        W_metric = torch.abs(current_weight) * torch.sqrt(scaler_row.reshape((1, -1)))
        W_mask = torch.ones_like(W_metric) == 1
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:, : int(W_metric.shape[1] * self.ratio)]
        W_mask.scatter_(1, indices, False)

        layer_id = id(layer)
        layer_name = [key for key, val in self.layers.items() if id(val) == layer_id][0]
        self.pruning_mask[layer_name] = W_mask
        self.pruning_indices[layer_name] = indices

    def ci(self, layer, inputs, outputs):
        current_weight = layer.weight.data
        X = inputs[0]
        batch_size = X.shape[0] // 2

        concern_inputs, non_concern_inputs = (
            X[:batch_size],
            X[batch_size:],
        )  # (batch_size, seq_dim, input_dim)

        calc_norm = lambda tensors, dim: torch.norm(
            tensors.reshape((-1, tensors.shape[-1])), dim=dim
        )

        new_shape = (1, -1)
        concern_norm = calc_norm(concern_inputs, dim=0).reshape(new_shape)
        non_concern_norm = calc_norm(non_concern_inputs, dim=0).reshape(new_shape)

        cosine_similarity = F.cosine_similarity(
            concern_inputs.reshape((-1, concern_inputs.shape[-1])),
            non_concern_inputs.reshape((-1, non_concern_inputs.shape[-1])),
            dim=0,
        ).reshape(1, -1)

        sine_similarity = torch.sqrt(1 - cosine_similarity**2)
        distance = torch.sqrt(concern_norm**2 + non_concern_norm**2)
        coefficient = (
            concern_norm
            + sine_similarity * torch.abs(concern_norm + non_concern_norm) / distance
        )

        importance_score = torch.abs(current_weight) * torch.abs(coefficient)

        indices_vector = None
        if self.method == "unstructed":
            sort_res = torch.sort(importance_score, dim=-1, stable=True)
            num_prune = int(current_weight.shape[1] * self.ratio)
            indices_matrix = sort_res[1][:, :num_prune]
            W_mask = (torch.ones_like(importance_score) == 1).scatter_(
                1, indices_matrix, False
            )
        elif self.method == "structed":
            importance_vector = torch.norm(importance_score, dim=1)
            num_prune = int(importance_vector.shape[0] * self.ratio)
            sort_res = torch.sort(importance_vector)
            indices_vector = sort_res[1][:num_prune]
            W_mask = (torch.ones_like(importance_vector) == 1).scatter_(
                0, indices_vector, False
            )
        else:
            raise NotImplementedError(f"{self.method} is not implemented")

        if self.method == "unstructed":
            sorted_indices_matrix = torch.sort(indices_matrix, dim=1)[0]
            indices = sorted_indices_matrix

        elif self.method == "structed":
            sorted_indices_vector = torch.sort(indices_vector)[0]
            indices = sorted_indices_vector
        else:
            raise NotImplementedError(f"The method {self.method} is not implemented")

        layer_id = id(layer)
        layer_name = [key for key, val in self.layers.items() if id(val) == layer_id][0]
        self.pruning_mask[layer_name] = W_mask
        self.pruning_indices[layer_name] = indices

    @staticmethod
    def apply(layer, method, axis, mask, keepdim):
        current_weight = layer.weight.data.clone()
        current_weight = current_weight * mask
        if not keepdim:
            if method == "structed":
                if axis == 0:
                    zero_rows = (current_weight == 0).all(dim=1)
                    current_weight = current_weight[~zero_rows]

                    if layer.bias is not None:
                        current_bias = layer.bias.data.clone()
                        layer.bias.data = current_bias[~zero_rows]
                elif axis == 1:
                    zero_cols = (current_weight == 0).all(dim=0)
                    current_weight = current_weight[:, ~zero_cols]
        layer.in_features = current_weight.shape[1]
        layer.out_features = current_weight.shape[0]
        layer.weight.data = current_weight


def find_layers(
    model: Module,
    layer_types: Optional[List[Type[Module]]] = None,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
    prefix: str = "",
) -> Dict[str, Module]:
    if layer_types is None:
        layer_types = [nn.Linear]
    if include_layers is None:
        include_layers = []
    if exclude_layers is None:
        exclude_layers = []
    layers_dict: Dict[str, Module] = {}

    def recursive_find(module: Module, prefix: str) -> None:
        for name, layer in module.named_children():
            layer_name = f"{prefix}.{name}" if prefix else name
            if any(exclude in layer_name for exclude in exclude_layers):
                continue
            if include_layers and not any(
                include in layer_name for include in include_layers
            ):
                if not any(isinstance(layer, t) for t in layer_types):
                    recursive_find(layer, layer_name)
                continue
            if isinstance(layer, tuple(layer_types)):
                layers_dict[layer_name] = layer
            else:
                recursive_find(layer, layer_name)

    recursive_find(model, prefix)

    return layers_dict


def get_hook(method):
    def hook(module, input, output):
        method(module, input, output)

    return hook


def prune_magnitude(
    model: Module,
    sparsity_ratio: float = 0.6,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
) -> None:
    layers = find_layers(
        model, include_layers=include_layers, exclude_layers=exclude_layers
    )
    for _, layer in layers.items():
        current_weight = layer.weight.data
        threshold = torch.sort(torch.abs(current_weight).flatten())[0][
            int(current_weight.numel() * sparsity_ratio)
        ]
        mask = torch.abs(current_weight) < threshold
        layer.weight.data[mask] = 0


def prune_norm_distribution(
    model: Module,
    sparsity_ratio: float = 0.4,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
) -> None:
    layers = find_layers(
        model, include_layers=include_layers, exclude_layers=exclude_layers
    )
    for _, layer in layers.items():
        current_weight = layer.weight.data
        mean = torch.mean(current_weight, dim=1, keepdim=True)
        std = torch.std(current_weight, dim=1, keepdim=True)
        z_scores = (current_weight - mean) / std

        lower_z, upper_z = norm.ppf(0.5 - sparsity_ratio / 2), norm.ppf(
            0.5 + sparsity_ratio / 2
        )
        mask = torch.logical_and(z_scores >= lower_z, z_scores < upper_z)
        layer.weight.data[mask] = 0


def prune_concern_identification(
    model: Module,
    config: Config,
    dominant_concern: SamplingDataset,
    non_dominant_concern: SamplingDataset,
    sparsity_ratio: float = 0.6,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
    method: str = "unstructed",
    keep_dim=True,
) -> None:
    layers = find_layers(
        model, include_layers=include_layers, exclude_layers=exclude_layers
    )
    handle_list = []
    pruner = Pruner(layers, ratio=sparsity_ratio, method=method)

    for name, layer in layers.items():
        if method == "structed":
            if "intermediate" in name:
                handle = layer.register_forward_hook(pruner.ci)
                handle_list.append(handle)
        else:
            handle = layer.register_forward_hook(pruner.ci)
            handle_list.append(handle)

    dominant_batches = list(dominant_concern)
    non_dominant_batches = list(non_dominant_concern)

    if len(dominant_batches) != len(non_dominant_batches):
        raise ValueError(
            "Batch sizes of dominant_concern and non_dominant_concern does not match."
        )

    combined_batches = {}
    keys = dominant_batches[0].keys()

    for key in keys:
        combined_batches[key] = torch.cat(
            [batch[key] for batch in dominant_batches + non_dominant_batches]
        )

    combined_dataloader = [combined_batches]
    propagate(model, combined_dataloader, config)
    for handle in handle_list:
        handle.remove()

    intermediate_mask = None
    for name, layer in layers.items():
        if method == "structed":
            if "intermediate" in name:
                current_mask = pruner.pruning_mask[name].to("cpu")
                intermediate_mask = current_mask
                current_mask = current_mask.unsqueeze(dim=1).expand(
                    -1, layer.weight.shape[1]
                )
                Pruner.apply(
                    layer,
                    method="structed",
                    axis=0,
                    mask=current_mask,
                    keepdim=keep_dim,
                )
            elif "output" in name:
                current_mask = intermediate_mask.unsqueeze(dim=0).expand(
                    layer.weight.shape[0], -1
                )
                Pruner.apply(
                    layer,
                    method="structed",
                    axis=1,
                    mask=current_mask,
                    keepdim=keep_dim,
                )
        elif method == "unstructed":
            current_mask = pruner.pruning_mask[name].to("cpu")
            Pruner.apply(
                layer, method="unstructed", axis=0, mask=current_mask, keepdim=keep_dim
            )


def prune_wanda(
    model: Module,
    config: Config,
    dataloader: SamplingDataset,
    sparsity_ratio: float = 0.4,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
):
    layers = find_layers(
        model, include_layers=include_layers, exclude_layers=exclude_layers
    )
    device = config.device
    handle_list = []

    pruner = Pruner(layers, ratio=sparsity_ratio, method="unstructed")
    for name, layer in layers.items():
        handle = layer.register_forward_hook(pruner.wanda)
        handle_list.append(handle)
    propagate(model, dataloader, config)

    for handle in handle_list:
        handle.remove()

    for name, layer in layers.items():
        current_mask = pruner.pruning_mask[name].to("cpu")
        Pruner.apply(
            layer, method="unstructed", axis=0, mask=current_mask, keepdim=True
        )
