import torch
import gc
import math
import numpy as np
import random
from tqdm.auto import tqdm
from src.utils.data_class import CustomEmbeddingDataset


def select_true_example(model, config, data_loader):
    num_labels = config.num_labels
    device = config.device

    sorted_inputs = []
    sorted_input_embeddings = []
    sorted_output_embeddings = []
    sorted_attention_mask = []

    correct_predictions = [[] for _ in range(num_labels)]
    model.eval()

    for batch in tqdm(data_loader, desc="Collecting embeddings", ascii=True):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                inputs, attention_mask=attention_mask, output_hidden_states=True
            )
        input_embeddings, output_embeddings = (
            outputs.hidden_states[0],
            outputs.hidden_states[-1],
        )
        predictions = outputs.logits.argmax(dim=-1)

        for i in range(len(labels)):
            if predictions[i] == labels[i]:
                correct_predictions[labels[i].item()].append(
                    (
                        outputs.logits[i, predictions[i]].item(),
                        input_embeddings[i].cpu().numpy(),
                        output_embeddings[i].cpu().numpy(),
                        attention_mask[i].cpu().numpy(),
                    )
                )

    for i in range(num_labels):
        correct_predictions[i].sort(key=lambda x: x[0], reverse=True)
        sorted_input_embeds = [pred[1] for pred in correct_predictions[i]]
        sorted_input_embeddings.append(sorted_input_embeds)

        sorted_output_embeds = [pred[2] for pred in correct_predictions[i]]
        sorted_output_embeddings.append(sorted_output_embeds)

        sorted_mask = [pred[3] for pred in correct_predictions[i]]
        sorted_attention_mask.append(sorted_mask)
        torch.cuda.empty_cache()

    return sorted_input_embeddings, sorted_output_embeddings, sorted_attention_mask


def extract_top_n_embeddings(extract_num, num_labels, embeds_list, mask_list):
    return_list1 = []
    return_list2 = []
    for i in range(extract_num):
        extracted_embeds = []
        extracted_mask = []
        for j in range(num_labels):
            class_embeds = np.array(embeds_list[j][i : i + 1])
            class_embeds_tensor = torch.tensor(class_embeds)
            class_mask = np.array(mask_list[j][i : i + 1])
            class_mask_tensor = torch.tensor(class_mask)
            extracted_embeds.append(class_embeds_tensor)
            extracted_mask.append(class_mask_tensor)
        return_list1.append(extracted_embeds)
        return_list2.append(extracted_mask)
    return return_list1, return_list2


def extract_bottom_n_embeddings(extract_num, num_labels, embeds_list, mask_list):
    return_list1 = []
    return_list2 = []
    for i in range(extract_num):
        extracted_embeds = []
        extracted_mask = []
        for j in range(num_labels):
            class_embeds = np.array(
                embeds_list[j][-(i + 1) : -i] or embeds_list[j][-(i + 1) :]
            )
            class_embeds_tensor = torch.tensor(class_embeds)
            class_mask = np.array(
                mask_list[j][-(i + 1) : -i] or mask_list[j][-(i + 1) :]
            )
            class_mask_tensor = torch.tensor(class_mask)
            extracted_embeds.append(class_embeds_tensor)
            extracted_mask.append(class_mask_tensor)
        return_list1.append(extracted_embeds)
        return_list2.append(extracted_mask)
    return return_list1, return_list2


def get_decimal_precision(value):
    str_value = str(value)
    if "." in str_value:
        return len(str_value.split(".")[1])
    else:
        return 0


def perturb(embeds, eps, grad):
    perturbed_embeds = embeds - eps * grad.sign()
    return perturbed_embeds


def calculate_gradient(model, input_embeds, attention_mask, target_class, device):
    input_embeds = input_embeds.clone().detach().requires_grad_(True).to(device)
    target = torch.tensor([target_class]).to(device)
    outputs = model(
        inputs_embeds=input_embeds, attention_mask=attention_mask, labels=target
    )
    init_pred = outputs.logits.argmax(dim=-1)
    loss = outputs.loss
    model.zero_grad()
    loss.backward()
    data_grad = input_embeds.grad
    return loss, data_grad


def fgsm_attack(
    model,
    source,
    target,
    input_embeds,
    attention_mask,
    start_eps,
    eps_step,
    max_eps,
    device,
):
    input_embeds = input_embeds.to(device)
    attention_mask = attention_mask.to(device)
    digits = get_decimal_precision(eps_step)
    targeted_eps = start_eps
    loss, data_grad = calculate_gradient(
        model, input_embeds, attention_mask, target, device
    )
    first_diff_eps = math.inf

    while targeted_eps <= max_eps:
        perturbed_embeds = perturb(input_embeds, targeted_eps, data_grad)
        adv_outputs = model(
            inputs_embeds=perturbed_embeds, attention_mask=attention_mask
        )
        adv_pred = adv_outputs.logits.argmax(dim=-1)

        if first_diff_eps == math.inf and adv_pred.item() != source:
            first_diff_eps = targeted_eps

        if adv_pred.item() == target:
            break
        else:
            targeted_eps += eps_step
            targeted_eps = round(targeted_eps, digits)

    return targeted_eps, first_diff_eps


def calculate_all_epsilon(
    model, config, logit_example, attention_mask, step_eps=0.01, max_eps=10.0
):
    num_labels = config.num_labels
    device = config.device

    first_changed_list = []
    targeted_list = []
    for i in range(num_labels):
        print(f"class {i}")
        targeted_temp = []
        first_changed_temp = []
        for j in range(num_labels):
            if i == j:
                targeted_temp.append((i, j, math.inf))
                first_changed_temp.append((i, j, math.inf))
                continue
            eps, first_diff_eps = fgsm_attack(
                model,
                i,
                j,
                logit_example[i],
                attention_mask[i],
                0.00,
                step_eps,
                max_eps,
                device,
            )
            if eps >= max_eps:
                targeted_temp.append((i, j, math.inf))
            else:
                targeted_temp.append((i, j, eps))
            first_changed_temp.append((i, j, first_diff_eps))
        targeted_list.append(targeted_temp)
        first_changed_list.append(first_changed_temp)

    flat_targeted_list = [
        (source, target, eps)
        for sublist in targeted_list
        for source, target, eps in sublist
        if eps != math.inf
    ]
    flat_first_changed_list = [
        (source, target, eps)
        for sublist in first_changed_list
        for source, target, eps in sublist
        if eps != math.inf and eps - (2 * step_eps) >= 0.00
    ]

    return flat_targeted_list, flat_first_changed_list


def generate_example(
    model,
    device,
    source,
    target,
    input_embeds,
    attention_mask,
    example_num,
    boundary_eps,
    step_eps,
):
    example_list = []
    example_label = []
    example_mask = []

    source_embedding = input_embeds[source].to(device)
    source_mask = attention_mask[source].to(device)

    loss, data_grad = calculate_gradient(
        model, source_embedding, source_mask, target, device
    )
    iter_num = 0
    while iter_num < example_num:
        eps = random.uniform(boundary_eps, boundary_eps + step_eps)
        generated_embeds = perturb(source_embedding, eps, data_grad)
        example_list.append(generated_embeds.cpu())
        example_mask.append(source_mask.cpu())
        example_label.append(source)
        iter_num += 1
        del generated_embeds
        torch.cuda.empty_cache()
    del source_embedding, loss, data_grad
    torch.cuda.empty_cache()
    return example_label, example_list, example_mask


def adjust_examples(
    example_list,
    example_label,
    example_mask,
    eps_list,
    embed_list,
    mask_list,
    target_num,
    step_eps,
    model,
    device,
):
    if len(example_list) < target_num:
        diff = target_num - len(example_list)
        for _ in range(diff):
            random_index = random.randint(0, len(embed_list) - 1)
            input_embed = embed_list[random_index]
            attention_mask = mask_list[random_index]
            eps_values = eps_list[random_index]
            if eps_values:
                source, target, eps = random.choice(eps_values)
                label, example, mask = generate_example(
                    model,
                    device,
                    source,
                    target,
                    input_embed,
                    attention_mask,
                    1,
                    eps - 2 * step_eps,
                    step_eps,
                )
                example_label.extend(label)
                example_list.extend(example)
                example_mask.extend(mask)
    elif len(example_list) > target_num:
        diff = len(example_list) - target_num
        for _ in range(diff):
            random_index = random.randint(0, len(example_list) - 1)
            example_list.pop(random_index)
            example_label.pop(random_index)
            example_mask.pop(random_index)


def make_example(
    model,
    config,
    data_loader,
    example_num,
    top_emb,
    bottom_emb,
    true_ratio,
    step_eps=0.01,
    max_eps=10.0,
):
    model = model.to(config.device)
    extract_embed_list = []
    extract_mask_list = []
    negative_eps = []
    positive_eps = []

    positive_example_list = []
    positive_example_label = []
    positive_mask_list = []

    negative_example_list = []
    negative_example_label = []
    negative_mask_list = []

    class_num = config.num_labels
    device = config.device

    positive_num = int(round(example_num * true_ratio))
    negative_num = example_num - positive_num
    extract_num = top_emb + bottom_emb

    per_emb_positive_example_num = int(round(positive_num / extract_num))
    per_emb_negative_example_num = int(round(negative_num / extract_num))

    input_embeds, _, attention_mask = select_true_example(model, config, data_loader)
    extracted_top_embed, extracted_top_mask = extract_top_n_embeddings(
        top_emb, class_num, input_embeds, attention_mask
    )
    extract_embed_list.extend(extracted_top_embed)
    extract_mask_list.extend(extracted_top_mask)
    extracted_bottom_embed, extracted_bottom_mask = extract_bottom_n_embeddings(
        bottom_emb, class_num, input_embeds, attention_mask
    )
    extract_embed_list.extend(extracted_bottom_embed)
    extract_mask_list.extend(extracted_bottom_mask)
    for i in range(len(extract_embed_list)):
        targeted_eps, first_changed_eps = calculate_all_epsilon(
            model,
            config,
            extract_embed_list[i],
            extract_mask_list[i],
            step_eps,
            max_eps,
        )
        positive_eps.append(first_changed_eps)
        negative_eps.append(targeted_eps)

    for i, input_embed in enumerate(extract_embed_list):
        for j, first_changed in enumerate(positive_eps[i]):
            per_positive_example_num = int(
                round(per_emb_positive_example_num / len(positive_eps[i]))
            )
            source, target, eps = first_changed
            pos_label, pos_example, pos_mask = generate_example(
                model,
                device,
                source,
                target,
                extract_embed_list[i],
                extract_mask_list[i],
                per_positive_example_num,
                eps - 2 * step_eps,
                step_eps,
            )
            positive_example_label.extend(pos_label)
            positive_example_list.extend(pos_example)
            positive_mask_list.extend(pos_mask)

        for j, targeted in enumerate(negative_eps[i]):
            per_negative_example_num = int(
                round(per_emb_negative_example_num / len(negative_eps[i]))
            )
            source, target, eps = targeted
            neg_label, neg_example, neg_mask = generate_example(
                model,
                device,
                source,
                target,
                extract_embed_list[i],
                extract_mask_list[i],
                per_negative_example_num,
                eps,
                step_eps,
            )
            negative_example_label.extend(neg_label)
            negative_example_list.extend(neg_example)
            negative_mask_list.extend(neg_mask)

    adjust_examples(
        positive_example_list,
        positive_example_label,
        positive_mask_list,
        positive_eps,
        extract_embed_list,
        extract_mask_list,
        positive_num,
        step_eps,
        model,
        device,
    )
    adjust_examples(
        negative_example_list,
        negative_example_label,
        negative_mask_list,
        negative_eps,
        extract_embed_list,
        extract_mask_list,
        negative_num,
        step_eps,
        model,
        device,
    )
    positive_example_list = [tensor.squeeze(0) for tensor in positive_example_list]
    positive_mask_list = [tensor.squeeze(0) for tensor in positive_mask_list]
    negative_example_list = [tensor.squeeze(0) for tensor in negative_example_list]
    negative_mask_list = [tensor.squeeze(0) for tensor in negative_mask_list]

    pos_embeddings = CustomEmbeddingDataset(
        {
            "embeddings": positive_example_list,
            "attention_mask": positive_mask_list,
            "labels": positive_example_label,
        }
    )
    neg_embeddings = CustomEmbeddingDataset(
        {
            "embeddings": negative_example_list,
            "labels": negative_example_label,
            "attention_mask": negative_mask_list,
        }
    )
    return pos_embeddings, neg_embeddings
