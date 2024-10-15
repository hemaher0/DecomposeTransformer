import os
import sys

sys.path.append("../")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import copy
import torch
from datetime import datetime
from ..src.utils.helper import Config, color_print
from ..src.utils.load import load_model, load_data, save_checkpoint
from ..src.models.evaluate import evaluate_model, get_sparsity, get_similarity
from ..src.utils.sampling import SamplingDataset
from ..src.pruning.prune_head import head_importance_prunning


def main():
    parser = argparse.ArgumentParser(description="Model Pruning and Evaluation")
    parser.add_argument("--name", type=str, default="Yahoo", help="Name of the dataset")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for computation"
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
        "--head_pruning_ratio",
        type=float,
        default=0.3,
        help="Sparsity ratio for Head pruning",
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
        default=["intermediate", "output"],
        help="Layers to include for pruning",
    )
    parser.add_argument(
        "--exclude_layers",
        type=str,
        nargs="+",
        default=["attention"],
        help="Layers to exclude for pruning",
    )

    args = parser.parse_args()

    name = args.name
    device = torch.device(args.device)
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_samples = args.num_samples
    head_pruning_ratio = args.head_pruning_ratio
    seed = args.seed

    color_print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

    config = Config(name, device)
    num_labels = config.num_labels

    model = load_model(config)

    train_dataloader, valid_dataloader, test_dataloader = load_data(
        config,
        batch_size=batch_size,
        num_workers=num_workers,
        do_cache=True,
    )

    color_print("Evaluate the original model")
    result = evaluate_model(model, config, test_dataloader)

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
    )

    module = copy.deepcopy(model)

    head_importance_prunning(module, config, positive_samples, head_pruning_ratio)

    color_print(f"Evaluate the pruned model {concern}")
    result = evaluate_model(module, config, test_dataloader)
    get_similarity(
        model,
        module,
        valid_dataloader,
        concern,
        num_samples,
        num_labels,
        device=device,
    )
    print(get_sparsity(module)[0])

    save_checkpoint(
        module, "results/", f"head_pruned_{name}_{head_pruning_ratio}p_class{concern}"
    )
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
