"""Server class for FedAvg."""
import numpy as np
import timeit
from model import ResNet18, SimSiam
import concurrent.futures
from logging import DEBUG, INFO
from typing import OrderedDict
import copy
import torch
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import (
    Callable,
    Dict,
    GetParametersIns,
    List,
    NDArrays,
    Optional,
    Tuple,
    Union,
)
from flwr.server import Server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.history import History
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]

from strategy_ssfl import HeteroSSFLStrategy

class FedAvgServer(Server):
    """Implement server for FedAvg."""

    def __init__(
        self,
        strategy: Strategy,
        num_classes: int,
        checkpoint_path: str = './models',
        client_manager: Optional[ClientManager] = None,
    ):
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.best_eval_acc = 0.0
        self.checkpoint_path = checkpoint_path
        self.len_params: int = 0

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        # save checkpoint 
        parameters_ndarrays = parameters_to_ndarrays(self.parameters)
        accuracy = float(metrics_aggregated["accuracy"])
        if accuracy > self.best_eval_acc:
            self.best_eval_acc = accuracy

            # Save model parameters and state
            if server_round == 0:
                # self.len_params = len(parameters_to_ndarrays(parameters_ndarrays))
                return None
            
            # List of keys for the arrays
            keys = [f'array{i+1}' for i in range(len(parameters_ndarrays))]

            np.savez(
                f"{self.checkpoint_path}.npz",  #eval_acc",
                **{key: arr for key, arr in zip(keys, parameters_ndarrays)},
                # arr_0 = parameters_ndarrays,
                scalar_0 = loss_aggregated,
                scalar_1 = self.best_eval_acc,
                scalar_2 = server_round
            )

            log(INFO, "Model saved with Best eval accuracy %.3f: ", self.best_eval_acc)
        return loss_aggregated, metrics_aggregated, (results, failures)

class HeteroSSFLServer(Server):
    """Implement server for Hetero SSFL."""

    def __init__(
        self,
        strategy: HeteroSSFLStrategy,
        num_classes: int,
        checkpoint_path: str = './models',
        client_manager: Optional[ClientManager] = None,
    ):
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        # self.model_params = ResNet18(num_classes)
        self.phi_K_aggregated = self.strategy.phi_K_aggregated
        self.len_params = self.strategy.len_params
        print(self.len_params)
        print(len(self.strategy.phi_K_aggregated))
        print(self.strategy.phi_K_aggregated[0].shape)
        # self.model_params = SimSiam(backbone=ResNet18(num_classes).resnet)
        # self.len_params = len(self.model_params.state_dict()) # len(model.state_dict())
        # self.model_params = None # memory econ 
        # self.best_eval_acc = 0.0
        # self.checkpoint_path = checkpoint_path

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters

    # pylint: disable=too-many-locals
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strateg
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=update_parameters_with_phi_K(self.parameters, self.phi_K_aggregated), #self.parameters
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar]] = (
            self.strategy.aggregate_fit(server_round, results, failures)
        )

        aggregated_result_arrays_combined = []
        if aggregated_result[0] is not None:
            aggregated_result_arrays_combined = parameters_to_ndarrays(
                aggregated_result[0]
            )
        updated_params = aggregated_result_arrays_combined[
            : self.len_params
        ]

        self.phi_K_aggregated = aggregated_result_arrays_combined[
            self.len_params :
        ]

        parameters_updated = ndarrays_to_parameters(updated_params)
        # metrics
        metrics_aggregated = aggregated_result[1]
        return parameters_updated, metrics_aggregated, (results, failures)

def update_parameters_with_phi_K(
    parameters: Parameters, phi_K: List[np.ndarray]
) -> Parameters:
    """Extend the list of parameters with the server control variate."""
    # extend the list of parameters arrays with the cv arrays
    # cv_np = [cv.numpy() for cv in s_cv]
    parameters_np = parameters_to_ndarrays(parameters)
    # .append
    parameters_np.extend(phi_K)
    return ndarrays_to_parameters(parameters_np)

def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)

def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res

def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)



# from collections import OrderedDict
# from omegaconf import DictConfig
# from hydra.utils import instantiate
# import torch
# from model import ResNet18, Net, test


# def get_on_fit_config(config: DictConfig):
#     """Return a function to configure the client's fit."""

#     def fit_config_fn(server_round: int):
#         """Return training configuration dict for each round.

#         Learning rate is reduced by a factor after set rounds.
#         """
#         config_res = {}
#         lr = config.optimizer.lr
#         if config.lr_scheduling:
#             if server_round == int(config.num_rounds / 2):
#                 lr = lr / 10

#             elif server_round == int(config.num_rounds * 0.75):
#                 lr = lr / 100
        
#         config_res["lr"] = lr
#         config_res["momentum"] = config.optimizer.momentum
#         # config_res["var_local_epochs"] = config.var_local_epochs
#         config_res["server_round"] = server_round
#         config_res["mu"] = config.optimizer.mu
#         config_res["weight_decay"] = config.optimizer.weight_decay
#         return config_res 
#         # return python dict
#     return fit_config_fn

# # def get_evaluate_fn(model_cfg: int, testloader)
# def get_evaluate_fn(num_classes: int, testloader):
#     """Return a function to evaluate the centralised global model."""

#     def evaluate_fn(server_round: int, parameters, config):
#         # model = instantiate(model_cfg)
#         model = ResNet18(num_classes)

#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         params_dict = zip(model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
#         model.load_state_dict(state_dict, strict=True)

#         loss, accuracy = test(model, testloader, device)

#         return loss, {"accuracy": accuracy}

#     return evaluate_fn