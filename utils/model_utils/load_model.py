import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from utils.helper import color_print


def load_model(model_config, checkpoint=None):
    cache_dir = model_config.cache_dir
    color_print(f"Loading the model.")
    model_config.summary()

    if model_config.task_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_name, cache_dir=cache_dir
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_config.model_name, cache_dir=cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name, cache_dir=cache_dir)

    # load check point
    if checkpoint is not None:
        load_checkpoint(model, model_config, checkpoint)
    model.to(model_config.device)
    color_print(f"The model {model_config.model_name} is loaded.")
    return model, tokenizer, checkpoint


def save_checkpoint(model, save_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, save_path)
    color_print(f"Checkpoint saved at {save_path}")


def load_checkpoint(model, model_config, checkpoint):
    checkpoint_path = os.path.join(model_config.Checkpoints, checkpoint)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path), map_location=model_config.device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            color_print(
                "Checkpoint structure is unrecognized. Check the keys or save format."
            )
    return model
