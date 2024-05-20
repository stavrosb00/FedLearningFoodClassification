from collections import OrderedDict
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from model import ResNet18, Net, test
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
from model import SimSiam
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter

class HeteroSSFLStrategy(FedAvg):
    """Implement custom strategy for HeteroSSFL with extra options and required aggregation functionality based on FedAvg class."""

    def __init__(
        self,
        num_classes: int = 10,
        checkpoint_path: str = './models',
        save_dir: str = "./clients",
        eval_every_n: int = 5,
        writer: SummaryWriter = None,
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
        self.model_params = SimSiam(backbone=ResNet18(num_classes, pretrained=False).resnet)
        self.len_params = len(self.model_params.state_dict()) # len(model.state_dict())
        self.model_params = None # memory econ 
        self.writer = writer
        # self.phi_K_aggregated = None

        # print(len(initial_parameters))
        nd_params = parameters_to_ndarrays(initial_parameters)
        state_params = ndarrays_to_parameters(nd_params[: self.len_params])
        self.phi_K_aggregated = nd_params[self.len_params :]
        self.initial_parameters = state_params
        nd_params = None
        state_params = None

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters
    
    # @property
    # def phi_K_aggregated(self) -> NDArrays:
    #     return self.phi_K_aggregated
    # def __getattribute__(self, phi_K_aggregated: NDArrays) -> NDArrays:
    #     return super().__getattribute__(phi_K_aggregated)

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

        combined_parameters_all_updates = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]

        num_examples_all_updates = [fit_res.num_examples for _, fit_res in results]
        # Zip models state parameters and num_examples
        parameters = [
            (update[: self.len_params], num_examples) #(update[: len_combined_parameter // 2], num_examples) 
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        # Aggregate parameters
        parameters_aggregated = aggregate(parameters)

        # Zip phi_Ks and num_examples
        phi_Ks = [
            (update[self.len_params :], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        # Aggregate phi_Ks parameters
        # either w_j = 1 for everyone -> K_mean either weighted aggregation based on num examples
        self.phi_K_aggregated = aggregate(phi_Ks)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            # keep CID->file, ServerRound->index, train values. Interpolation the progress between ticks in second place
            for _, m in fit_metrics:
                # "loss": train_loss, "d_loss": d_loss, "cka_loss": cka_loss,
                temp_dict = {"client_id" : m["client_id"], "server_round": server_round, "loss": m["loss"], "d_loss": m["d_loss"], "cka_loss": m["cka_loss"], "fit_mins": m["fit_mins"]} #, "accuracy": m["accuracy"], 
                # self.writer.add_scalars(f"client_fit_progress_{m['client_id']}/loss", m["loss"], server_round)
                self.writer.add_scalars(f"client_fit_progress_{m['client_id']}/losses", 
                                        {"loss": m["loss"], "d_loss": m["d_loss"], "cka_loss": m["cka_loss"]}, 
                                        server_round)
                self.writer.add_scalar(f"client_fit_progress_{m['client_id']}/fit_mins", m["fit_mins"], server_round)
                if os.path.exists(f"{self.dir}/client_fit_progress_{m['client_id']}.csv"):
                    temp_df = pd.DataFrame(temp_dict, index=[0])
                    # update by appending only the new values without the header on the .csv
                    temp_df.to_csv(f"{self.dir}/client_fit_progress_{m['client_id']}.csv", mode='a', index=False, header=False)
                else:
                    # init progress file for first time
                    temp_df = pd.DataFrame(temp_dict, index=[0])
                    temp_df.to_csv(f"{self.dir}/client_fit_progress_{m['client_id']}.csv", index=False)

            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            # self.writer.add_scalar(f"server/weighted_loss", metrics_aggregated['loss'], server_round) 
            self.writer.add_scalars(f"server/weighted_losses", 
                                    {"loss": metrics_aggregated['loss'], "d_loss": metrics_aggregated['d_loss'], "cka_loss": metrics_aggregated['cka_loss']}, server_round) 
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # ndarrays_to_parameters(server_parameters_aggregated + cv_parameters_reg_summed + server_buffers_aggregated),
        return (
            ndarrays_to_parameters(parameters_aggregated + self.phi_K_aggregated),
            metrics_aggregated,
        )

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
        # save checkpoint 
        accuracy = float(metrics["accuracy"])
        self.writer.add_scalar(f"server/kNN_accuracy", accuracy, server_round)
        if accuracy > self.best_test_acc:
            self.best_test_acc = accuracy

            # Save model parameters and state
            if server_round == 0:
                return None
            
            # List of keys for the arrays
            keys = [f'array{i+1}' for i in range(len(parameters_ndarrays))]

            np.savez(
                f"{self.checkpoint_path}.npz",  #test_acc",
                **{key: arr for key, arr in zip(keys, parameters_ndarrays)},
                # arr_0 = parameters_ndarrays,
                scalar_0 = loss,
                scalar_1 = self.best_test_acc,
                scalar_2 = server_round
            )

            log(INFO, "Model saved with Best Test kNN accuracy %.3f: ", self.best_test_acc)
        return loss, metrics
    # def aggregate_evaluate(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[ClientProxy, EvaluateRes]],
    #     failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    # ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    #     """Aggregate evaluation losses using weighted average."""
    #     if not results:
    #         return None, {}
    #     # Do not aggregate if there are failures and failures are not accepted
    #     if not self.accept_failures and failures:
    #         return None, {}

    #     # Aggregate loss
    #     loss_aggregated = weighted_loss_avg(
    #         [
    #             (evaluate_res.num_examples, evaluate_res.loss)
    #             for _, evaluate_res in results
    #         ]
    #     )

    #     # Aggregate custom metrics if aggregation fn was provided
    #     metrics_aggregated = {}
    #     if self.evaluate_metrics_aggregation_fn:
    #         eval_metrics = [(res.num_examples, res.metrics) for _, res in results]

    #         for _, m in eval_metrics:
    #             temp_dict = {"client_id" : m['client_id'], "server_round": server_round, "loss": m["loss"], "accuracy": m["accuracy"]}
    #             if os.path.exists(f"{self.dir}/client_eval_progress_{m['client_id']}.csv"):
    #                 temp_df = pd.DataFrame(temp_dict, index=[0])
    #                 # update by appending only the new values without the header on the .csv
    #                 temp_df.to_csv(f"{self.dir}/client_eval_progress_{m['client_id']}.csv", mode='a', index=False, header=False)
    #             else:
    #                 # init progress file for first time
    #                 temp_df = pd.DataFrame(temp_dict, index=[0])
    #                 temp_df.to_csv(f"{self.dir}/client_eval_progress_{m['client_id']}.csv", index=False)

    #         metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
    #     elif server_round == 1:  # Only log this warning once
    #         log(WARNING, "No evaluate_metrics_aggregation_fn provided")

    #     return loss_aggregated, metrics_aggregated


