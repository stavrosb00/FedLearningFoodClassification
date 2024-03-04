"""Server class for SCAFFOLD."""
import numpy as np
from model import ResNet18
import concurrent.futures
from logging import DEBUG, INFO
from typing import OrderedDict

import torch
from flwr.common import (
    Code,
    FitIns,
    FitRes,
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
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

class ScaffoldServer(Server):
    """Implement server for SCAFFOLD."""

    """Implement server for SCAFFOLD."""

    def __init__(
        self,
        strategy: Strategy,
        num_classes: int,
        client_manager: Optional[ClientManager] = None,
    ):
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.model_params = ResNet18(num_classes)
        self.server_cv: List[torch.Tensor] = []

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        # self.strategy.
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
        # parameters_to_ndarrays(get_parameters_res.parameters)
        # print(f"params init len{len(parameters_to_ndarrays(get_parameters_res.parameters))}") #122
        # exit(0)
        # print(f"params init len{get_parameters_res.parameters}")
        # Server control variate init
        # get_parameters_res.
        self.server_cv = [
            torch.from_numpy(t)
            for t in parameters_to_ndarrays(get_parameters_res.parameters)
        ]
        # print(f"len server cv init {len(self.server_cv)}") #122
        # exit()
        # print(len(self.server_cv))
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
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=update_parameters_with_cv(self.parameters, self.server_cv),
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
        aggregated_result: Tuple[
            Optional[Parameters], Dict[str, Scalar]
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        aggregated_result_arrays_combined = []
        if aggregated_result[0] is not None:
            aggregated_result_arrays_combined = parameters_to_ndarrays(
                aggregated_result[0]
            )
        aggregated_parameters = aggregated_result_arrays_combined[
            : len(aggregated_result_arrays_combined) // 2
        ]
        aggregated_cv_update = aggregated_result_arrays_combined[
            len(aggregated_result_arrays_combined) // 2 :
        ]

        # convert server cv into ndarrays
        server_cv_np = [cv.numpy() for cv in self.server_cv]
        # update server cv
        total_clients = len(self._client_manager.all())
        # fraction of participant clients
        cv_multiplier = len(results) / total_clients
        self.server_cv = [
            torch.from_numpy(cv + cv_multiplier * aggregated_cv_update[i])
            for i, cv in enumerate(server_cv_np)
        ]

        # update parameters x = x + 1* aggregated_update, /eta_g = 1
        curr_params = parameters_to_ndarrays(self.parameters)
        updated_params = [
            x + aggregated_parameters[i] for i, x in enumerate(curr_params)
        ]
        parameters_updated = ndarrays_to_parameters(updated_params)

        # metrics
        metrics_aggregated = aggregated_result[1]
        return parameters_updated, metrics_aggregated, (results, failures)


def update_parameters_with_cv(
    parameters: Parameters, s_cv: List[torch.Tensor]
) -> Parameters:
    """Extend the list of parameters with the server control variate."""
    # extend the list of parameters arrays with the cv arrays
    cv_np = [cv.numpy() for cv in s_cv]
    parameters_np = parameters_to_ndarrays(parameters)
    # parameters_np = np.hstack([parameters_np, cv_np])
    parameters_np.extend(cv_np)
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