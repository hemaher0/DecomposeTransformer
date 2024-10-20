import copy
import torch
import numpy as np
from typing import *
import torch.nn as nn
from functools import partial
from transformers.pytorch_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)


def calculate_head_importance(
    model,
    config,
    data,
    normalize_scores_by_layer=True,
):
    device = config.device
    from functools import partial

    gradients = {}
    context_layers = {}

    def save_grad(gradients, layer_idx, grad):
        gradients[f"context_layer_{layer_idx}"] = grad

    def forward_hook(module, input, output, gradients, context_layers, layer_idx):
        context_layers[f"context_layer_{layer_idx}"] = output[0]
        output[0].register_hook(partial(save_grad, gradients, layer_idx))

    def reshape(tensors, shape, num_heads):
        batch_size = shape[0]
        seq_len = shape[1]
        head_dim = shape[2] // num_heads
        tensors = tensors.reshape(batch_size, seq_len, num_heads, head_dim)
        tensors = tensors.permute(0, 2, 1, 3)
        return tensors

    forward_handles = []

    for layer_idx in range(model.bert.config.num_hidden_layers):
        self_att = model.bert.encoder.layer[layer_idx].attention.self
        handle = self_att.register_forward_hook(
            partial(
                forward_hook,
                gradients=gradients,
                context_layers=context_layers,
                layer_idx=layer_idx,
            )
        )
        forward_handles.append(handle)

    """Calculate head importance scores"""
    # Disable dropout
    model.eval()
    # Device
    device = device or next(model.parameters()).device

    # Prepare data loader
    # Head importance tensor
    n_layers = model.bert.config.num_hidden_layers
    n_heads = model.bert.config.num_attention_heads
    head_dim = model.bert.config.hidden_size // n_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    tot_tokens = 0
    first_batch = next(iter(data))
    is_embeds = "embeddings" in first_batch
    for step, batch in enumerate(data):
        if is_embeds:
            embeddings = batch["embeddings"].to(device)
        else:
            input_ids = batch["input_ids"].to(device)
        input_mask = batch["attention_mask"].to(device)
        label_ids = batch["labels"].to(device)
        # Compute gradients
        if is_embeds:
            loss = model(
                inputs_embeds=embeddings, attention_mask=input_mask, labels=label_ids
            ).loss
        else:
            loss = model(input_ids, attention_mask=input_mask, labels=label_ids).loss
        loss.backward()

        for layer_idx in range(model.bert.config.num_hidden_layers):
            ctx = context_layers[f"context_layer_{layer_idx}"]
            grad_ctx = gradients[f"context_layer_{layer_idx}"]
            shape = ctx.shape
            ctx = reshape(ctx, shape, n_heads)
            grad_ctx = reshape(ctx, shape, n_heads)

            # Take the dot
            dot = torch.einsum("bhli,bhli->bhl", [grad_ctx, ctx])
            head_importance[layer_idx] += dot.abs().sum(-1).sum(0).detach()
            del ctx, grad_ctx, dot

        tot_tokens += input_mask.float().detach().sum().data

    head_importance[:-1] /= tot_tokens
    # Layerwise importance normalization
    if normalize_scores_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(
            torch.pow(head_importance, exponent).sum(-1), 1 / exponent
        )
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    # for layer_idx in range(model.bert.config.num_hidden_layers):
    #     head_importance[layer_idx] *= torch.norm(model.bert.encoder.layer[layer_idx].attention.self.value.weight)

    for layer_idx in range(model.bert.config.num_hidden_layers):
        for head in range(n_heads):
            start_idx = head * head_dim
            end_idx = (head + 1) * head_dim
            value_weight_norm = torch.norm(
                model.bert.encoder.layer[layer_idx].attention.self.value.weight[
                    :, start_idx:end_idx
                ]
            )
            # head_importance[layer_idx] += value_weight_norm.detach()
            head_importance[layer_idx][start_idx:end_idx] += torch.min(
                head_importance[layer_idx]
            )
            head_importance[layer_idx][start_idx:end_idx] /= torch.max(
                head_importance[layer_idx]
            ) + torch.min(head_importance[layer_idx])

    for handle in forward_handles:
        handle.remove()
    return head_importance


def head_importance_prunning(
    model, config, dominant_concern, sparsity_ratio, method="unstructed", scheduler=None
):
    num_attention_heads = model.config.num_attention_heads
    num_hidden_layers = model.config.num_hidden_layers
    model = model.to(config.device)
    total_heads_to_prune = int(num_attention_heads * num_hidden_layers * sparsity_ratio)

    total_heads_to_prune = max(total_heads_to_prune, num_hidden_layers)
    print(f"Total heads to prune: {total_heads_to_prune}")
    pruned_heads = set()

    if scheduler is not None:
        steps = scheduler.get_steps()
    else:
        steps = [1.0]

    for step_ratio in steps:
        heads_to_prune = int(total_heads_to_prune * step_ratio)

        head_importance_list = calculate_head_importance(
            model, config, dominant_concern
        )
        head_importance_list = head_importance_list.cpu()
        print(head_importance_list)

        if method == "unstructed":
            sorted_indices = torch.argsort(head_importance_list.view(-1))
            prune_list = [
                (int(idx // num_attention_heads), int(idx % num_attention_heads))
                for idx in sorted_indices[:heads_to_prune]
            ]
        elif method == "structed":
            heads_per_layer = heads_to_prune // num_hidden_layers
            prune_list = []
            for layer_idx in range(num_hidden_layers):
                sorted_heads = torch.argsort(head_importance_list[layer_idx])
                prune_list.extend(
                    [
                        (layer_idx, head.item())
                        for head in sorted_heads[:heads_per_layer]
                    ]
                )

        for layer_index, head_index in prune_list:
            if (layer_index, head_index) not in pruned_heads:
                prune_heads(
                    model.bert.encoder.layer[layer_index].attention,
                    [head_index],
                    method=method,
                )
                pruned_heads.add((layer_index, head_index))
    print(pruned_heads)


def prune_heads(layer, heads, method):
    if len(heads) == 0:
        return
    heads, index = find_pruneable_heads_and_indices(
        heads,
        layer.self.num_attention_heads,
        layer.self.attention_head_size,
        layer.pruned_heads,
    )

    # if method == "unstructed":
    layer.self.query = zero_out_head_weights(
        layer.self.query, heads, layer.self.attention_head_size
    )
    layer.self.key = zero_out_head_weights(
        layer.self.key, heads, layer.self.attention_head_size
    )
    layer.self.value = zero_out_head_weights(
        layer.self.value, heads, layer.self.attention_head_size
    )
    layer.output.dense = zero_out_head_weights(
        layer.output.dense, heads, layer.self.attention_head_size, dim=1
    )
    # elif method == "structed":
    #     layer.self.query = prune_linear_layer(layer.self.query, index)
    #     layer.self.key = prune_linear_layer(layer.self.key, index)
    #     layer.self.value = prune_linear_layer(layer.self.value, index)
    #     layer.output.dense = prune_linear_layer(layer.output.dense, index)

    #     layer.self.num_attention_heads = layer.self.num_attention_heads - len(heads)
    #     layer.self.all_head_size = layer.self.attention_head_size *  layer.self.num_attention_heads
    #     layer.pruned_heads = layer.pruned_heads.union(heads)


def zero_out_head_weights(
    layer: nn.Linear, heads: Set[int], head_size: int, dim: int = 0
) -> nn.Linear:
    """
    Zero out the weights of the specified heads in the linear layer.

    Args:
        layer (`torch.nn.Linear`): The layer to modify.
        heads (`Set[int]`): The indices of heads to zero out.
        head_size (`int`): The size of each head.
        dim (`int`, *optional*, defaults to 0): The dimension on which to zero out the weights.

    Returns:
        `torch.nn.Linear`: The modified layer with weights of specified heads zeroed out.
    """
    for head in heads:
        start_index = head * head_size
        end_index = (head + 1) * head_size
        if dim == 0:
            layer.weight.data[start_index:end_index] = 0
        elif dim == 1:
            layer.weight.data[:, start_index:end_index] = 0

    return layer
