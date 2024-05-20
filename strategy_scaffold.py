"""SCAFFOLD strategy"""

from functools import reduce
from logging import WARNING

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitIns,
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
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from logging import INFO
from model import ResNet18
import pandas as pd
import os


class ScaffoldStrategy(FedAvg):
    """Implement custom strategy for SCAFFOLD based on FedAvg class."""

    def __init__(
        self,
        num_classes: int = 10,
        checkpoint_path: str = './models',
        save_dir: str = "./clients",
        # eval_every_n: int = 5,
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
        self.best_test_acc = 0.0
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        # server learning rate of delta_x
        # self.eta_g: int = 1
        self.model_params = ResNet18(self.num_classes)
        self.len_params = len([p for p in self.model_params.parameters()])
        self.model_params = None
        # self.eval_every_n = eval_every_n #global eval freq
    
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
        # print(len(combined_parameters_all_updates)) #->5 clients sampled
        # S = len(results)
        # len_combined_parameter = len(combined_parameters_all_updates[0])
        # print(len_combined_parameter) # 184
        
        # print(f"len combined strat[0]: {len_combined_parameter}") # 62+62 -> 124
        # norm_summed_params = np.sum(combined_parameters_all_updates, axis=0) / len(results)
        # end_params=[]
        # voodoo
        # summed_params_list = [(for param in cid_param_list) for cid_param_list in zip(combined_parameters_all_updates)]
        num_examples_all_updates = [fit_res.num_examples for _, fit_res in results]
        # Zip parameters and num_examples
        update_dx = [
            (update[: self.len_params], num_examples) #(update[: len_combined_parameter // 2], num_examples) 
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        # Aggregate parameters
        server_parameters_aggregated = aggregate(update_dx)

        # Zip buffers and num_examples
        update_buff = [
            (update[2 * self.len_params :], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        # Aggregate buffers parameters
        server_buffers_aggregated = aggregate(update_buff)

        # Zip client_cv_updates and num_examples
        client_cv_updates_and_num_examples = [
            (update[self.len_params : 2*self.len_params], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        aggregated_cv_update = aggregate(client_cv_updates_and_num_examples)



        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            # keep CID->file, ServerRound->index, train values. Interpolation the progress between ticks in second place
            for _, m in fit_metrics:
                temp_dict = {"client_id" : m["client_id"], "server_round": server_round, "loss": m["loss"], "accuracy": m["accuracy"], "fit_mins": m["fit_mins"]}
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

        # ndarrays_to_parameters(server_parameters_aggregated + cv_parameters_reg_summed + server_buffers_aggregated),
        return (
            ndarrays_to_parameters(server_parameters_aggregated + aggregated_cv_update + server_buffers_aggregated),
            metrics_aggregated,
        )
    
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
                temp_dict = {"client_id" : m["client_id"], "server_round": server_round, "loss": m["loss"], "accuracy": m["accuracy"]}
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
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

        if eval_res is None:
            return None

        loss, metrics = eval_res
        # # save checkpoint 
        # accuracy = float(metrics["accuracy"])
        # if accuracy > self.best_test_acc:
        #     self.best_test_acc = accuracy

        #     # Save model parameters and state
        #     if server_round == 0:
        #         return None
            
        #     # List of keys for the arrays
        #     keys = [f'array{i+1}' for i in range(len(parameters_ndarrays))]

        #     np.savez(
        #         f"{self.checkpoint_path}.npz",  #test_acc",
        #         **{key: arr for key, arr in zip(keys, parameters_ndarrays)},
        #         # arr_0 = parameters_ndarrays,
        #         scalar_0 = loss,
        #         scalar_1 = self.best_test_acc,
        #         scalar_2 = server_round
        #     )

        #     log(INFO, "Model saved with Best Test accuracy %.3f: ", self.best_test_acc)

        return loss, metrics
    
#temp trash 
    
        # server_updated_params = [update[: len_combined_parameter // 2] for update in combined_parameters_all_updates]
        # # Reg(div /S) sum delta server for each layer of parameters
        # layers = [
        #     [layer for layer in layers] for layers in server_updated_params
        #     ]
        # server_parameters_reg_summed: NDArrays = [
        #     reduce(np.add, layer_updates) / S
        #     for layer_updates in zip(*layers)
        # ]
    
    # eta_g = 1
        # eta_g = float(np.sqrt(len(results)))
        # cv_parameters_reg_summed: NDArrays = [
        #     reduce(np.add, layer_updates) * cv_multiplier#/ self.N #/ S
        #     for layer_updates in zip(*layers)
        # ]
        # for upd_param, aggr_param in zip(updated_params[i], aggregated_parameters[ct_p]):
                #     upd_param = upd_param + eta_g * aggr_param
    

    # Reg(div /S) sum delta client updates for each layer of parameters
        # cv_updated_params = [
        #     update[self.len_params : 2*self.len_params] #update[len_combined_parameter // 2 :]
        #     for update in combined_parameters_all_updates] 
        # # Create a list of layers
        # layers = [
        #     [layer for layer in layers] for layers in cv_updated_params
        #     ]
        # # Compute average of each layer  (S / N)*(sum(Delta_c) / S) = sum(Delta_c) / N 
        # cv_multiplier = self.eta_g / self.min_available_clients
        
        # cv_parameters_reg_summed: NDArrays = [
        #     reduce(np.add, layer_updates) * cv_multiplier#/ self.N #/ S
        #     for layer_updates in zip(*layers)
        # ]

class ScaffoldStrategyV2(FedAvg):
    """Implement custom strategy for SCAFFOLD version2 based on FedAvg class."""
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
        self.checkpoint_path = checkpoint_path
        self.eval_every_n = eval_every_n #global eval freq
        # self.best_test_acc = 0.0
        # self.num_classes = num_classes
        # self.model_params = ResNet18(self.num_classes)
        # # server learning rate of delta_x
        # # self.eta_g: int = 1
        # self.len_params = len([p for p in self.model_params.parameters()])
        # self.model_params = None
        

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
        len_combined_parameter = len(combined_parameters_all_updates[0])
        num_examples_all_updates = [fit_res.num_examples for _, fit_res in results]
        # Zip parameters and num_examples
        weights_results = [
            (update[: len_combined_parameter // 2], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        # Aggregate parameters
        parameters_aggregated = aggregate(weights_results)

        # Zip client_cv_updates and num_examples
        client_cv_updates_and_num_examples = [
            (update[len_combined_parameter // 2 :], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        aggregated_cv_update = aggregate(client_cv_updates_and_num_examples)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            # keep CID->file, ServerRound->index, train values. Interpolation the progress between ticks in second place
            for _, m in fit_metrics:
                temp_dict = {"client_id" : m["client_id"], "server_round": server_round, "loss": m["loss"], "accuracy": m["accuracy"], "fit_mins": m["fit_mins"]}
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

        return (
            ndarrays_to_parameters(parameters_aggregated + aggregated_cv_update),
            metrics_aggregated,
        )
    
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
                temp_dict = {"client_id" : m["client_id"], "server_round": server_round, "loss": m["loss"], "accuracy": m["accuracy"]}
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
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

        if eval_res is None:
            return None

        loss, metrics = eval_res
        # save checkpoint 
        # accuracy = float(metrics["accuracy"])
        # if accuracy > self.best_test_acc:
        #     self.best_test_acc = accuracy

        #     # Save model parameters and state
        #     if server_round == 0:
        #         return None
            
        #     # List of keys for the arrays
        #     keys = [f'array{i+1}' for i in range(len(parameters_ndarrays))]

        #     np.savez(
        #         f"{self.checkpoint_path}.npz",  #test_acc",
        #         **{key: arr for key, arr in zip(keys, parameters_ndarrays)},
        #         # arr_0 = parameters_ndarrays,
        #         scalar_0 = loss,
        #         scalar_1 = self.best_test_acc,
        #         scalar_2 = server_round
        #     )

        #     log(INFO, "Model saved with Best Test accuracy %.3f: ", self.best_test_acc)

        return loss, metrics
    