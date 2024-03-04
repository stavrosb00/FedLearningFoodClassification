from collections import OrderedDict
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from model import ResNet18, Net, test
from typing import Dict, List, Tuple
from flwr.common.typing import Metrics
import numpy as np

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Return weighted average of accuracy metrics as evaluation."""
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate (weighted average)
        loss = np.sum(losses) / np.sum(examples) 
        accuracy = np.sum(accuracies) / np.sum(examples) 
        # Extra metrics for accuracy progress
        accs = np.array([m["accuracy"] for _, m in metrics])
        diff_acc = accs - accuracy
        #for balanced datasets mean_diff_acc=0 as expected
        mean_diff_acc = np.mean(diff_acc)
        # "mean_diff_acc": mean_diff_acc
        #variance for statistical accuracy comparison (equal variances)
        # var_acc = np.var(accs)
        var_diff_acc = np.var(diff_acc)
        #return custom metrics
        return {"loss": loss, "accuracy": accuracy, "mean_diff_acc": mean_diff_acc,"var_diff_acc": var_diff_acc}

def get_on_fit_config(config: DictConfig):
    """Return a function to configure the client's fit."""

    def fit_config_fn(server_round: int):
        """Return training configuration dict for each round.

        Learning rate is reduced by a factor after set rounds.
        """
        config_res = {}
        lr = config.optimizer.lr
        if config.lr_scheduling:
            if server_round == int(config.num_rounds / 2):
                lr = lr / 10

            elif server_round == int(config.num_rounds * 0.75):
                lr = lr / 100
        
        config_res["lr"] = lr
        config_res["server_round"] = server_round
        # config_res["momentum"] = config.optimizer.momentum
        # config_res["var_local_epochs"] = config.var_local_epochs
        # config_res["mu"] = config.optimizer.mu
        # config_res["weight_decay"] = config.optimizer.weight_decay
        return config_res 
    
    return fit_config_fn

# def get_evaluate_fn(model_cfg: int, testloader)
def get_evaluate_fn_scaffold(num_classes: int, testloader):
    """Return a function to evaluate the centralised global model."""

    def evaluate_fn(server_round: int, parameters, config):
        # model = instantiate(model_cfg)
        model = ResNet18(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        keys = [k for k, _ in model.named_parameters()]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)

        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn

def get_evaluate_fn(num_classes: int, testloader):
    """Return a function to evaluate the centralised global model."""

    def evaluate_fn(server_round: int, parameters, config):
        # model = instantiate(model_cfg)
        model = ResNet18(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn