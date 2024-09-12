# In[]: Import Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from typing import *
from torch.nn import Module
import numpy as np

from utils.prune_utils.prune import find_layers, propagate
from utils.dataset_utils.sampling import SamplingDataset
from utils.model_utils.CKA import linear_CKA, kernel_CKA
from utils.model_utils.cca_core import get_cca_similarity


def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).sum().item() / len(labels)
    return {"accuracy": accuracy}


def evaluate_model(
    model, model_config, test_dataloader, is_binary=False
):
    model.eval()
    input_ids = None
    embeddings = None
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    loss_fn = nn.CrossEntropyLoss()
    model = model.to(model_config.device)

    for idx, batch in enumerate(
        tqdm(test_dataloader, desc="Evaluating the model", dynamic_ncols=True)
    ):
        is_embeds = "embeddings" in batch

        if not is_embeds:
            input_ids = batch["input_ids"].to(model_config.device)
            batch_size = input_ids.size(0)
        else:
            embeddings = batch["embeddings"].to(model_config.device)
            batch_size = embeddings.size(0)

        attention_mask = batch["attention_mask"].to(model_config.device)
        labels = batch["labels"]
        
        if isinstance(labels, int):
            labels = torch.tensor([labels]).to(model_config.device)
        else:
            labels = labels.to(model_config.device)
        
        total_samples += batch_size

        with torch.no_grad():
            if not is_embeds:
                outputs = model(input_ids, attention_mask=attention_mask)
                del input_ids
            else:
                outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
                del embeddings
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * batch_size
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        del attention_mask, labels, logits, outputs, loss, pred
        torch.cuda.empty_cache()

    avg_loss = total_loss / total_samples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary" if is_binary else "macro",
        zero_division=0,
    )

    report = classification_report(all_labels, all_preds, zero_division=0)
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
    return overall_sparsity, sparsity_dict


def similar(
    model: Module,
    module: Module,
    dataloader: DataLoader,
    concern: int,
    num_samples,
    num_labels,
    device: torch.device = torch.device("cpu"),
    seed=44,
) -> None:
    positive_samples = SamplingDataset(
        dataloader,
        concern,
        num_samples,
        num_labels,
        True,
        4,
        device=device,
        resample=False,
        seed=seed,
    )
    negative_samples = SamplingDataset(
        dataloader,
        concern,
        num_samples,
        num_labels,
        False,
        4,
        device=device,
        resample=False,
        seed=seed,
    )
    concern_outputs1 = propagate(model, positive_samples, device)
    concern_outputs2 = propagate(module, positive_samples, device)
    non_concern_outputs1 = propagate(model, negative_samples, device)
    non_concern_outputs2 = propagate(module, negative_samples, device)

    hidden_states = lambda x, y: (
        x.reshape(-1, x.shape[-1]).T,
        y.reshape(-1, y.shape[-1]).T,
    )

    h1, h2 = hidden_states(concern_outputs1, concern_outputs2)
    h3, h4 = hidden_states(non_concern_outputs1, non_concern_outputs2)
    cca_results_concern = get_cca_similarity(h1, h2, epsilon=1e-6)
    cca_results_non_concern = get_cca_similarity(h3, h4, epsilon=1e-6)
    print(f"CCA coefficients mean concern: {cca_results_concern['mean']}")
    print(f"CCA coefficients mean non-concern: {cca_results_non_concern['mean']}")
    print(f"Linear CKA concern: {linear_CKA(h1, h2)}")
    print(f"Linear CKA non-concern: {linear_CKA(h3, h4)}")
    print(f"Kernel CKA concern: {kernel_CKA(h1, h2)}")
    print(f"Kernel CKA non-concern: {kernel_CKA(h3, h4)}")
