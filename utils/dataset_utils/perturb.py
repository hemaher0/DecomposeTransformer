import torch
import gc
from tqdm.auto import tqdm
from utils.dataset_utils.load_dataset import (
    Embedding
)

def extract_top_k_embeddings(model_config, extract_num, embeddings, attention_mask):
    num_labels = model_config.num_labels
    extracted_embeddings_tensor = torch.stack([embeddings[i][:extract_num] for i in range(num_labels)])
    extracted_attention_mask_tensor = torch.stack([attention_mask[i][:extract_num] for i in range(num_labels)])
    
    return extracted_embeddings_tensor, extracted_attention_mask_tensor
  
def get_decimal_precision(value):
    str_value = str(value)
    if '.' in str_value:
        return len(str_value.split('.')[1])
    else:
        return 0
      
def perturb(embeddings, eps, grad):
    perturbed_embeddings = embeddings - eps*grad.sign()
    return perturbed_embeddings
  
  
def calculate_gradient(model, input_embeddings, attention_mask, target_class, device):
    input_embeddings = input_embeddings.to(device)
    attention_mask = attention_mask.to(device)
    model.eval()
    batch_size = input_embeddings.size(0)
    losses = []
    gradients = []

    for i in range(batch_size):
        embedding = input_embeddings[i:i+1].requires_grad_(True)
        mask = attention_mask[i:i+1]
        target = torch.tensor([target_class], device=device)
        
        outputs = model(inputs_embeds=embedding, attention_mask=mask, labels=target)
        loss = outputs.loss
        
        model.zero_grad()
        loss.backward()
        losses.append(loss.item())
        gradients.append(embedding.grad.squeeze(0).detach())
        torch.cuda.empty_cache()
        gc.collect()
    
    stacked_gradients = torch.stack(gradients)
    return losses, stacked_gradients
  
def fgsm_attack(model, model_config, target_class, input_embeddings, attention_mask, start_epsilon, epsilon_step, max_epsilon):
    device = model_config.device
    model.eval()
    input_embeddings = input_embeddings.to(device)
    attention_mask = attention_mask.to(device)
    digits = get_decimal_precision(epsilon_step)
    eps = start_epsilon
    loss, data_grad = calculate_gradient(model, input_embeddings, attention_mask,target_class, device)
    
    while eps <= max_epsilon:
        perturbed_embeddings = perturb(input_embeddings, eps, data_grad)
        with torch.no_grad():
            adv_outputs = model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask)
            adv_pred = adv_outputs.logits.argmax(dim=-1)
        
        if adv_pred.item() == target_class:
            break
        else:
            eps += epsilon_step
            eps = round(eps, digits)
    return eps, perturbed_embeddings
  
def calculate_all_epsilon(model, model_config, top_k_embeddings, attention_mask, step_epsilon = 0.01, max_epsilon = 10.0):
    num_labels = model_config.num_labels
    epsilon_list = []
    max_eps = max_epsilon
    batch_size = top_k_embeddings[0].shape[0]
    for i in tqdm(range(num_labels), desc="Calculating all epsilons", dynamic_ncols=True):
        temp = []
        
        for j in range(num_labels):
            if i == j:
                temp.append([float("inf")] * batch_size)
                continue
            epsilon_per_batch = []
            for b in range(batch_size):
                embedding = top_k_embeddings[i][b:b+1]
                mask = attention_mask[i][b:b+1]
                epsilon, perterbed = fgsm_attack(
                    model, model_config, j, embedding, mask, 0.00, step_epsilon, max_eps
                )
                if epsilon >= max_eps:
                    epsilon_per_batch.append(float("inf"))
                else:
                    epsilon_per_batch.append(epsilon)
            temp.append(epsilon_per_batch)
        epsilon_list.append(temp)
    return epsilon_list
  
def select_true_example(model, model_config, data_loader):
    num_labels = model_config.num_labels
    device = model_config.device
    correct_predictions = [[] for _ in range(num_labels)]
    model.eval()
    
    for batch in tqdm(data_loader, desc="Collecting correct predictions", dynamic_ncols=True):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask, output_hidden_states=True)
            input_embeddings, output_embeddings = outputs.hidden_states[0], outputs.hidden_states[-1]

        predictions = outputs.logits.argmax(dim=-1)
        correct_indices = (predictions == labels).nonzero(as_tuple=True)[0]
        
        for i in correct_indices:
            label = labels[i].item()
            logits = outputs.logits[i, predictions[i]].item()
            
            correct_predictions[label].append((
                logits,
                inputs[i].cpu(),
                input_embeddings[i].cpu(),
                output_embeddings[i].cpu(),
                attention_mask[i].cpu()
            ))
        del inputs, labels, attention_mask, outputs, input_embeddings, output_embeddings, predictions, correct_indices
        torch.cuda.empty_cache()
                
    sorted_inputs = []
    sorted_input_embeddings = []
    sorted_output_embeddings = []
    sorted_attention_mask = []
    
    for predictions in correct_predictions:
        predictions.sort(key=lambda x: x[0], reverse=True)
        
        sorted_inputs.append(torch.stack([pred[1] for pred in predictions]))
        sorted_input_embeddings.append(torch.stack([pred[2] for pred in predictions]))
        sorted_output_embeddings.append(torch.stack([pred[3] for pred in predictions]))
        sorted_attention_mask.append(torch.stack([pred[4] for pred in predictions]))
    
    return sorted_inputs, sorted_input_embeddings, sorted_output_embeddings, sorted_attention_mask
  
def generate_example(model, model_config, start_epsilon, end_epsilon, source_class, target_class, input_embeddings, attention_mask, per_attack_example_num, min_step_eps=1e-5):
    device = model_config.device
    example_list = []
    example_label = []
    attention_mask_list = []
    
    source_embedding = input_embeddings[source_class].to(device)
    source_attention_mask = attention_mask[source_class].to(device)
    loss, data_grad = calculate_gradient(model, source_embedding, source_attention_mask, target_class, device)

    step_eps = (end_epsilon - start_epsilon) / per_attack_example_num
    step_eps = max(step_eps, min_step_eps)
    iter_eps = start_epsilon
    iter_num = 0
    
    while iter_eps < end_epsilon and iter_num < per_attack_example_num:
        generated_embeddings = perturb(source_embedding, iter_eps, data_grad)
        batch_size = generated_embeddings.shape[0]
        for b in range(batch_size):
            example_list.append(generated_embeddings[b].cpu())
            attention_mask_list.append(source_attention_mask[b].cpu())
        example_label.extend([source_class] * batch_size)
        
        iter_eps += step_eps
        iter_num += 1
        
        del generated_embeddings
        torch.cuda.empty_cache()
        
    del source_embedding, source_attention_mask, loss, data_grad
    torch.cuda.empty_cache()
    
    return example_label, example_list, attention_mask_list
  
def make_example(model, model_config, data_loader, example_num, emb_num, class_num, true_ratio, step_epsilon=0.01, max_epsilon=10.0):
    device = model_config.device
    positive_example_list = []
    positive_example_label = []
    positive_attention_mask = []
    
    negative_example_list = []
    negative_example_label = []
    negative_attention_mask = []
    
    positive_num = int(example_num * true_ratio)
    negative_num = example_num - positive_num
    
    print("positive num : ", positive_num)
    print("negative num : ", negative_num)
    
    inputs, input_embeddings, output_embeddings, attention_mask = select_true_example(model, model_config, data_loader)
    extracted_embeddings, extracted_attention_mask = extract_top_k_embeddings(model_config, emb_num, input_embeddings, attention_mask)
    
    epsilon_list = calculate_all_epsilon(model, model_config, extracted_embeddings, extracted_attention_mask, step_epsilon, max_epsilon)
    
    per_class_positive_example_num = int(positive_num / class_num)
    per_class_negative_example_num = int(negative_num / class_num)
    print("per_class_positive_example_num : ", per_class_positive_example_num)
    print("per_class_negative_example_num : ", per_class_negative_example_num)
    
    for source_class in range(class_num):
        inf_num = sum(all(item == float('inf') for item in epsilon) for epsilon in epsilon_list[source_class])
        per_target_positive_example_num = int(per_class_positive_example_num / (class_num - inf_num))
        per_target_negative_example_num = int(per_class_negative_example_num / (class_num - inf_num))
        
        for target_class in range(class_num):
            epsilon = epsilon_list[source_class][target_class]

            if all(eps == float("inf") for eps in epsilon):
                continue

            for eps in epsilon:
                if eps == float("inf"):
                    continue
                    
                pos_label, pos_examples, pos_mask = generate_example(
                    model,
                    model_config,
                    start_epsilon=0.00,
                    end_epsilon=eps - step_epsilon,
                    source_class=source_class,
                    target_class=target_class,
                    input_embeddings=extracted_embeddings,
                    attention_mask=extracted_attention_mask,
                    per_attack_example_num=per_target_positive_example_num
                )
                positive_example_label.extend(pos_label)
                positive_example_list.extend(pos_examples)
                positive_attention_mask.extend(pos_mask)
                
                neg_label, neg_examples, neg_mask = generate_example(
                    model,
                    model_config,
                    start_epsilon=eps,
                    end_epsilon=2 * eps - step_epsilon,
                    source_class=source_class,
                    target_class=target_class,
                    input_embeddings=extracted_embeddings,
                    attention_mask=extracted_attention_mask,
                    per_attack_example_num=per_target_negative_example_num
                )
                negative_example_label.extend(neg_label)
                negative_example_list.extend(neg_examples)
                negative_attention_mask.extend(neg_mask)

        del pos_label, pos_examples, pos_mask, neg_label, neg_examples, neg_mask
        torch.cuda.empty_cache()
        gc.collect()
        
    pos_embeddings = Embedding(positive_example_list, positive_example_label, positive_attention_mask)
    neg_embeddings = Embedding(negative_example_list, negative_example_label, negative_attention_mask)

    del positive_example_list, positive_example_label, positive_attention_mask
    del negative_example_list, negative_example_label, negative_attention_mask
    torch.cuda.empty_cache()
    gc.collect()
    
    return pos_embeddings, neg_embeddings