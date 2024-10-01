import os
import sys

sys.path.append("../")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import copy
import torch
from datetime import datetime
from utils.helper import ModelConfig, color_print
from utils.dataset_utils.load_dataset import (
    load_data,
)
from utils.model_utils.save_module import save_module
from utils.model_utils.load_model import load_model
from utils.model_utils.evaluate import evaluate_model, get_sparsity, similar
from utils.dataset_utils.sampling import SamplingDataset
from utils.prune_utils.prune import prune_concern_identification


def main():
    parser = argparse.ArgumentParser(description="Model Pruning and Evaluation")
    parser.add_argument("--name", type=str, default="Yahoo", help="Name of the dataset")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for computation"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to load model from"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for data loaders"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loaders"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="Number of samples for sampling dataset",
    )
    parser.add_argument(
        "--concern", type=int, default=0, help="Target Concern for decompose"
    )
    parser.add_argument(
        "--ci_ratio", type=float, default=0.3, help="Sparsity ratio for CI"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--include_layers",
        type=str,
        nargs="+",
        default=["attention", "intermediate", "output"],
        help="Layers to include for pruning",
    )
    parser.add_argument(
        "--exclude_layers",
        type=str,
        nargs="+",
        default=None,
        help="Layers to exclude for pruning",
    )

    args = parser.parse_args()

    name = args.name
    device = torch.device(args.device)
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_samples = args.num_samples
    ci_ratio = args.ci_ratio
    seed = args.seed
    include_layers = args.include_layers
    exclude_layers = args.exclude_layers

    color_print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

    model_config = ModelConfig(name, device)
    num_labels = model_config.num_labels

    model, tokenizer, checkpoint = load_model(model_config)

    train_dataloader, valid_dataloader, test_dataloader = load_data(
        model_config.dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        do_cache=True,
        seed=seed,
    )

    color_print("Evaluate the original model")
    result = evaluate_model(model, model_config, test_dataloader)

    concern = args.concern
    color_print("#Module " + str(concern) + " in progress....")

    positive_samples = SamplingDataset(
        train_dataloader,
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
        train_dataloader,
        concern,
        num_samples,
        num_labels,
        False,
        4,
        device=device,
        resample=False,
        seed=seed,
    )

    module = copy.deepcopy(model)

    prune_concern_identification(
        module,
        model_config,
        positive_samples,
        negative_samples,
        include_layers=include_layers,
        exclude_layers=exclude_layers,
        sparsity_ratio=ci_ratio,
    )

    color_print(f"Evaluate the pruned model {concern}")
    result = evaluate_model(module, model_config, test_dataloader)
    similar(
        model,
        module,
        valid_dataloader,
        concern,
        num_samples,
        num_labels,
        device=device,
        seed=seed,
    )
    print(get_sparsity(module)[0])

    # save_module(module, "Modules/", f"ci_{name}_{ci_ratio}p_class{concern}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
