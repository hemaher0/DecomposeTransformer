import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from typing import *
from torch.nn import Module

from .CKA import linear_CKA, kernel_CKA
from .cca_core import get_cca_similarity
from ..pruning.propagate import (
    propagate_embeddings,
    propagate_image_classifier,
    propagate_text_classifier,
)
from ..pruning.prune import find_layers, propagate
from ..utils.helper import Config
from ..utils.sampling import SamplingDataset


def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).sum().item() / len(labels)
    return {"accuracy": accuracy}


def evaluate_model(model, config, test_dataloader, is_binary=False, verbose=False):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    device = config.device

    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    task_type = config.task_type

    for idx, batch in enumerate(
        tqdm(
            test_dataloader, desc="Evaluating the model", dynamic_ncols=True, ascii=True
        )
    ):
        is_embeds = "embeddings" in batch

        if task_type == "text_classification":
            if is_embeds:
                outputs, labels = propagate_embeddings(model, batch, device)
            else:
                outputs, labels = propagate_text_classifier(model, batch, device)
        elif task_type == "image_classification":
            outputs, labels = propagate_image_classifier(model, batch, device)
        batch_size = labels.size(0)
        total_samples += batch_size

        with torch.no_grad():

            logits = (
                outputs.get("logits")
                if isinstance(outputs, dict)
                else getattr(outputs, "logits", outputs)
            )

            loss = loss_fn(logits, labels)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * batch_size
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        del labels, logits, outputs, loss, pred
        torch.cuda.empty_cache()

    avg_loss = total_loss / total_samples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary" if is_binary else "macro",
        zero_division=0,
    )

    report = classification_report(all_labels, all_preds, zero_division=0, digits=4)
    if not verbose:
        print(f"Loss: {avg_loss:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        print(report)

    return {
        "loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "report": report,
    }


def calculate_sparsity(param):
    return (param == 0).sum().item() / param.numel()


def get_perplexity(model, dataloader: DataLoader, config: Config):
    model.eval()
    total_loss = 0
    total_words = 0
    task_type = config.task_type
    device = config.device

    for batch in dataloader:
        is_embeds = "embeddings" in batch

        if task_type == "text_classification":
            if is_embeds:
                outputs, labels = propagate_embeddings(model, batch, device)
            else:
                outputs, labels = propagate_text_classifier(model, batch, device)
        elif task_type == "image_classification":
            outputs, labels = propagate_image_classifier(model, batch, device)

        logits = (
            outputs.get("logits")
            if isinstance(outputs, dict)
            else getattr(outputs, "logits", outputs)
        )

        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        total_loss += loss.item()
        total_words += labels.numel()

    avg_loss = total_loss / total_words
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(perplexity.item())
    return perplexity.item()


def get_sparsity(model, layer_types=None, include_layers=None, exclude_layers=None):
    layers = find_layers(
        model,
        layer_types=layer_types,
        include_layers=include_layers,
        exclude_layers=exclude_layers,
    )
    sparsity_dict = {}
    total_sparsity = 0
    total_params = 0

    for name, module in layers.items():
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                sparsity = calculate_sparsity(param.data)
                sparsity_dict[f"{name}.{param_name}"] = sparsity
                total_sparsity += sparsity * param.numel()
                total_params += param.numel()

    overall_sparsity = total_sparsity / total_params if total_params != 0 else 0
    print(overall_sparsity)
    print(sparsity)
    return overall_sparsity, sparsity_dict


def get_similarity(
    model: Module,
    module: Module,
    dataloader: DataLoader,
    concern: int,
    num_samples,
    config: Config,
) -> None:
    config.init_seed()

    positive_samples = SamplingDataset(
        dataloader,
        config,
        concern,
        num_samples,
        True,
        4,
        resample=False,
    )
    negative_samples = SamplingDataset(
        dataloader,
        config,
        concern,
        num_samples,
        False,
        4,
        resample=False,
    )
    concern_outputs1 = propagate(model, positive_samples, config)
    concern_outputs2 = propagate(module, positive_samples, config)
    non_concern_outputs1 = propagate(model, negative_samples, config)
    non_concern_outputs2 = propagate(module, negative_samples, config)

    hidden_states = lambda x, y: (
        x.reshape(-1, x.shape[-1]).T,
        y.reshape(-1, y.shape[-1]).T,
    )

    h1, h2 = hidden_states(concern_outputs1, concern_outputs2)
    h3, h4 = hidden_states(non_concern_outputs1, non_concern_outputs2)
    cca_results_concern = get_cca_similarity(h1, h2, epsilon=1e-6)
    cca_results_non_concern = get_cca_similarity(h3, h4, epsilon=1e-6)

    cca_concern_mean = cca_results_concern["mean"][0]
    cca_non_concern_mean = cca_results_non_concern["mean"][0]

    print(f"CCA coefficients mean concern: {cca_concern_mean}")
    print(f"CCA coefficients mean non-concern: {cca_non_concern_mean}")

    linear_cka_concern = linear_CKA(h1, h2)
    linear_cka_non_concern = linear_CKA(h3, h4)
    kernel_cka_concern = kernel_CKA(h1, h2)
    kernel_cka_non_concern = kernel_CKA(h3, h4)

    print(f"Linear CKA concern: {linear_cka_concern}")
    print(f"Linear CKA non-concern: {linear_cka_non_concern}")
    print(f"Kernel CKA concern: {kernel_cka_concern}")
    print(f"Kernel CKA non-concern: {kernel_cka_non_concern}")
