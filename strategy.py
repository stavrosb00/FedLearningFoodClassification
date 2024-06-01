from collections import OrderedDict
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from model import ResNet18, SimSiam, Net, test, knn_monitor
from typing import Dict, List, Tuple
from flwr.common.typing import Metrics
import numpy as np

from flwr.server.strategy import FedAvg
from functools import reduce
from logging import WARNING

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from logging import INFO
import pandas as pd
import os
import math

class CustomFedAvgStrategy(FedAvg):
    """Implement custom strategy for FedAvg with extra options based on FedAvg class."""

    def __init__(
        self,
        num_classes: int = 10,
        checkpoint_path: str = './models',
        save_dir: str = "./clients",
        eval_every_n: int = 5,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__(
        fraction_fit = fraction_fit,
        fraction_evaluate = fraction_evaluate,
        min_fit_clients = min_fit_clients,
        min_evaluate_clients = min_evaluate_clients,
        min_available_clients = min_available_clients,
        evaluate_fn = evaluate_fn,
        on_fit_config_fn = on_fit_config_fn,
        on_evaluate_config_fn = on_evaluate_config_fn,
        accept_failures = accept_failures,
        initial_parameters = initial_parameters,
        fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,
        inplace = inplace)
        
        self.dir = save_dir
        self.num_classes = num_classes 
        self.checkpoint_path = checkpoint_path
        self.best_test_acc = 0.0
        self.eval_every_n = eval_every_n
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]

            # keep CID->file, ServerRound->index, train values. Interpolation the progress between ticks in second place
            for _, m in fit_metrics:
                temp_dict = {"client_id" : m['client_id'], "server_round": server_round, "loss": m["loss"], "accuracy": m["accuracy"], "fit_mins": m["fit_mins"]}
                if os.path.exists(f"{self.dir}/client_fit_progress_{m['client_id']}.csv"):
                    temp_df = pd.DataFrame(temp_dict, index=[0])
                    # update by appending only the new values without the header on the .csv
                    temp_df.to_csv(f"{self.dir}/client_fit_progress_{m['client_id']}.csv", mode='a', index=False, header=False)
                else:
                    # init progress file for first time
                    temp_df = pd.DataFrame(temp_dict, index=[0])
                    temp_df.to_csv(f"{self.dir}/client_fit_progress_{m['client_id']}.csv", index=False)

            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]

            for _, m in eval_metrics:
                temp_dict = {"client_id" : m['client_id'], "server_round": server_round, "loss": m["loss"], "accuracy": m["accuracy"]}
                if os.path.exists(f"{self.dir}/client_eval_progress_{m['client_id']}.csv"):
                    temp_df = pd.DataFrame(temp_dict, index=[0])
                    # update by appending only the new values without the header on the .csv
                    temp_df.to_csv(f"{self.dir}/client_eval_progress_{m['client_id']}.csv", mode='a', index=False, header=False)
                else:
                    # init progress file for first time
                    temp_df = pd.DataFrame(temp_dict, index=[0])
                    temp_df.to_csv(f"{self.dir}/client_eval_progress_{m['client_id']}.csv", index=False)

            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Overide default evaluate method to save model parameters."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        # if (server_round % self.eval_every_n ==0):
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

        if eval_res is None:
            return None

        loss, metrics = eval_res
    #     # save checkpoint 
    #     accuracy = float(metrics["accuracy"])
    #     if accuracy > self.best_test_acc:
    #         self.best_test_acc = accuracy

    #         # Save model parameters and state
    #         if server_round == 0:
    #             return None
            
    #         # List of keys for the arrays
    #         keys = [f'array{i+1}' for i in range(len(parameters_ndarrays))]

    #         np.savez(
    #             f"{self.checkpoint_path}.npz",  #test_acc",
    #             **{key: arr for key, arr in zip(keys, parameters_ndarrays)},
    #             # arr_0 = parameters_ndarrays,
    #             scalar_0 = loss,
    #             scalar_1 = self.best_test_acc,
    #             scalar_2 = server_round
    #         )

    #         log(INFO, "Model saved with Best Test accuracy %.3f: ", self.best_test_acc)
    # else:
    #     print(f"Only global testing every {self.eval_every_n} rounds...")
    #     return None
        return loss, metrics

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

def weighted_average_ssfl(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Return weighted average of metrics as evaluation method for SSFL."""
    # Multiply accuracy of each client by number of examples used and weighted aggregate them
    # accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    loss = np.sum(losses) / np.sum(examples) 
    result = {"loss": loss}
    if any("d_loss" in m for _, m in metrics):
        d_losses = [num_examples * m["d_loss"] for num_examples, m in metrics]
        d_loss = np.sum(d_losses) / np.sum(examples)
        result["d_loss"] = d_loss
    
    if any("cka_loss" in m for _, m in metrics):
        cka_loss = [num_examples * m["cka_loss"] for num_examples, m in metrics]
        cka_loss = np.sum(cka_loss) / np.sum(examples)
        result["cka_loss"] = cka_loss
    #return custom metrics
    return result #, "accuracy": accuracy}



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

def get_on_fit_config_ssfl(config: DictConfig):
    """Return a function to configure the client's fit."""

    def fit_config_fn(server_round: int):
        """Return training configuration dict for each round.

        Learning rate is reduced by a factor after set rounds.
        """  
        config_res = {}
        base_lr = config.optimizer.lr
        # Optional - Cosine decay rule w/ warmup depending on server rounds level
        if config.cos_decay:
            warm_up_rounds = config.warm_up_rounds # 5 me 10
            init_lr = base_lr * config.batch_size / 256
            n_rounds=800
            # linear spacing of learning rate during warm-up rounds
            if server_round < warm_up_rounds:
                cur_lr = np.linspace(0, init_lr, warm_up_rounds)[server_round]
            # cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * server_round / config.num_rounds)) 
            # after warm-up
            cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (150 + server_round - warm_up_rounds) / (n_rounds - warm_up_rounds))) 
            # print(cur_lr)
        else:
            cur_lr = base_lr # 0.01 0.0075 h 0.03
        # lr = adjust_learning_rate()
        config_res["lr"] = cur_lr
        config_res["server_round"] = server_round
        # isws return "loc_epochs" kai "local_scheduling" = [...linspace kok...]
        # config_res["num_rounds"] = config.num_rounds
        return config_res 
    
    return fit_config_fn

def get_evaluate_fn_ssfl(num_classes: int, testloader, memoryloader):
    """Return a function to evaluate the centralised global SimSiam SSFL model."""
    def evaluate_fn(server_round: int, parameters, config):
        # model = SimSiam(backbone=ResNet18(num_classes).resnet)
        model = SimSiam(backbone=ResNet18(num_classes, pretrained=False).resnet)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cuda:1")
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Backbone ResNet + projector MLP -> representations => Syndyasmo + Linear Evaluation head  #BYOL 
        # Backbone -> representations => Sketo backbone + Linear evaluation head
        accuracy = knn_monitor(model.encoder.to(device), memoryloader, testloader, k=min(25, len(memoryloader.dataset)), device=device, hide_progress=True)  #min(25, 7500)
        # loss, accuracy = test(model, testloader, device)
        loss = 0 # could be None also
        return loss, {"accuracy": accuracy}
    return evaluate_fn
    

# def get_evaluate_fn(model_cfg: int, testloader)
def get_evaluate_fn_scaffold(num_classes: int, testloader):
    """Return a function to evaluate the centralised global model."""

    def evaluate_fn(server_round: int, parameters, config):
        model = ResNet18(num_classes)
        # print(f"eval fn parms {len(parameters)}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        # keys = [k for k, _ in model.named_parameters()]
        # params_dict = zip(keys, parameters)
        # state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        # model.load_state_dict(state_dict, strict=False)

        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn

def get_evaluate_fn(num_classes: int, testloader):
    """Return a function to evaluate the centralised global model."""

    def evaluate_fn(server_round: int, parameters, config):
        # model = instantiate(model_cfg)
        model = ResNet18(num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn