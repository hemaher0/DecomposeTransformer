import torch
from transformers import AutoConfig
from transformers import BertForSequenceClassification


class ConcernModularizationBert:
    @staticmethod
    def channeling(module, active_node, dead_node, concern_idx, device):
        weight = module.classifier.weight
        bias = module.classifier.bias

        active_top1 = max(active_node)
        dead_top1 = max(dead_node)

        _active = [idx for idx, val in enumerate(active_node) if val >= active_top1 / 2]
        _dead = [
            idx
            for idx, val in enumerate(dead_node)
            if val >= dead_top1 / 2 and idx != concern_idx
        ]

        active = list(set(_active))
        dead = list(set(_dead))

        print(f"dead node \n{dead}")
        print(f"active node \n{active}")

        inter1 = [val if idx in dead else 0 for idx, val in enumerate(dead_node)]
        inter2 = [val if idx in active else 0 for idx, val in enumerate(active_node)]

        print(f"weight factor \n{inter1}")
        print(f"weight factor \n{inter2}")

        inter = torch.tensor([inter1, inter2], dtype=torch.float32).to(device)

        norms = inter.norm(p=2, dim=1, keepdim=True)
        inter_normalized = inter / norms

        new_weight = torch.matmul(inter_normalized, weight)
        new_bias = torch.matmul(inter_normalized, bias.unsqueeze(1)).squeeze(1)

        # set_parameters(module.classifier, new_weight, new_bias)
        module.classifier.out_features = 2

    @staticmethod
    def convert2binary(model_config, ref_model):
        config = AutoConfig.from_pretrained(model_config.model_name)
        config.id2label = {0: "negative", 1: "positive"}
        config.label2id = {"negative": 0, "positive": 1}
        config._num_labels = 2
        module = BertForSequenceClassification(config)
        module = module.to(model_config.device)
        module.load_state_dict(ref_model.state_dict())
        return module
