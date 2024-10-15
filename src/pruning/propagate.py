import gc
import torch


def propagate_text_classifier(model, batch, device, output_hidden_states=False):
    model = model.to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"]

    if isinstance(labels, int):
        labels = torch.tensor([labels]).to(device)
    else:
        labels = labels.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

    model = model.to("cpu")
    return outputs, labels


def propagate_image_classifier(model, batch, device, output_hidden_states=False):
    model = model.to(device)
    images = batch["image"].float().to(device)
    labels = batch["labels"]

    if isinstance(labels, int):
        labels = torch.tensor([labels]).to(device)
    else:
        labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images, output_hidden_states=output_hidden_states)

    model = model.to("cpu")
    return outputs, labels


def propagate_embeddings(model, batch, device, output_hidden_states=False):
    model = model.to(device)
    embeddings = batch["embeddings"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"]

    if isinstance(labels, int):
        labels = torch.tensor([labels]).to(device)
    else:
        labels = labels.to(device)

    with torch.no_grad():
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

    model = model.to("cpu")
    return outputs, labels


def propagate(model, dataloader, config, chunk_size=4):
    all_outputs = []
    chunk_outputs = []

    device = config.device
    task_type = config.task_type
    model = model.to(device)
    model.eval()

    for batch in dataloader:
        is_embeds = "embeddings" in batch

        if task_type == "text_classification":
            if is_embeds:
                outputs, _ = propagate_embeddings(
                    model, batch, device, output_hidden_states=True
                )
            else:
                outputs, _ = propagate_text_classifier(
                    model, batch, device, output_hidden_states=True
                )
            chunk_outputs.append(outputs.hidden_states[-1])
        elif task_type == "image_classification":
            outputs, _ = propagate_image_classifier(
                model, batch, device, output_hidden_states=True
            )
            chunk_outputs.append(outputs["hidden_states"][-1])
        if len(chunk_outputs) == chunk_size:
            all_outputs.append(torch.cat(chunk_outputs))
            chunk_outputs = []
    if len(chunk_outputs) > 0:
        all_outputs.append(torch.cat(chunk_outputs))
    all_outputs = torch.cat(all_outputs).cpu().detach().numpy()
    torch.cuda.empty_cache()
    gc.collect()
    return all_outputs
    # all_outputs = []

    # device = config.device
    # task_type = config.task_type
    # model = model.to(device)
    # model.eval()

    # for batch in dataloader:
    #     is_embeds = "embeddings" in batch
    #     chunks = {k: torch.chunk(v, chunk_size, dim=0) for k, v in batch.items()}

    #     num_chunks = len(next(iter(chunks.values())))
    #     for idx in range(num_chunks):
    #         chunked_batch = {k: v[idx] for k, v in chunks.items()}

    #         if task_type == "text_classification":
    #             if is_embeds:
    #                 outputs, _ = propagate_embeddings(
    #                     model, chunked_batch, device, output_hidden_states=True
    #                 )
    #             else:
    #                 outputs, _ = propagate_text_classifier(
    #                     model, chunked_batch, device, output_hidden_states=True
    #                 )
    #             all_outputs.append(outputs.hidden_states[-1])
    #         elif task_type == "image_classification":
    #             outputs, _ = propagate_image_classifier(
    #                 model, chunked_batch, device, output_hidden_states=True
    #             )
    #             all_outputs.append(outputs["hidden_states"][-1])

    # all_outputs = torch.cat(all_outputs).cpu().detach().numpy()
    # torch.cuda.empty_cache()
    # gc.collect()
    # return all_outputs
