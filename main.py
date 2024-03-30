import pickle
from pathlib import Path
import time
import os
import pandas as pd
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from hydra.utils import call, instantiate
from dataset import load_dataset
from client import generate_client_fn
# from server import get_on_fit_config, get_evaluate_fn
from strategy import get_on_fit_config, get_evaluate_fn, get_evaluate_fn_scaffold, weighted_average, CustomFedAvgStrategy #get_metrics_aggregation_fn
from utils import save_results_as_pickle, plot_metric_from_history
from strategy_scaffold import ScaffoldStrategy
from client_scaffold import ScaffoldClient
from server_scaffold import ScaffoldServer


# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    start = time.time()
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    ## 2. Prepare your dataset
    trainloaders, validationloaders, testloader = load_dataset(cfg.datapath, cfg.subset, cfg.num_classes, cfg.num_workers, 
                                                               cfg.num_clients, cfg.batch_size, cfg.partitioning, cfg.alpha, cfg.balance, cfg.seed, cfg.val_ratio)
    # for i in range(4):
    #     print(len(trainloaders[i]))
    #     print(len(validationloaders[i].dataset))
    #     print(len(trainloaders[i].dataset))
    
    # print(len(testloader))
    # print(testloader.batch_size)
    # print(len(testloader.dataset))
    # return 0

    save_path = HydraConfig.get().runtime.output_dir
    ## 3. Define your clients
    if cfg.strategy.client_fn._target_ == "client_scaffold.generate_client_fn":
        
        # client_progress = os.path.join(save_path, "client_progress")
        # print("Local progress for clients who participated in rounds are saved to: ", client_progress)
        client_progress = os.path.join(save_path, "clients")
        print("Local progress and client variances for scaffold clients are saved to: ", client_progress)
        client_fn = call(cfg.strategy.client_fn, trainloaders, validationloaders, cfg.num_classes, 
                         cfg.local_epochs, cfg, save_dir=client_progress,
        )
        evaluate_fn = get_evaluate_fn_scaffold(cfg.num_classes, testloader)
    else:
        client_progress = os.path.join(save_path, "clients")
        print("Local progress for clients who participated in rounds are saved to: ", client_progress)
        client_fn = call(cfg.strategy.client_fn, trainloaders, validationloaders, cfg.num_classes, cfg.local_epochs, cfg, save_dir=client_progress)
        evaluate_fn = get_evaluate_fn(cfg.num_classes, testloader)

    # client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes, cfg.local_epochs, cfg)

    ## 4. Define your strategy
    # in the strategy's `aggregate_fit()` method
    # You can implement a custom strategy to have full control on all aspects including: how the clients are sampled,
    # how updated models from the clients are aggregated, how the model is evaluated on the server, etc
    # To control how many clients are sampled, strategies often use a combination of two parameters `fraction_{}` and `min_{}_clients`
    # where `{}` can be either `fit` or `evaluate`, depending on the FL stage. The final number of clients sampled is given by the formula
    # ``` # an equivalent bit of code is used by the strategies' num_fit_clients() and num_evaluate_clients() built-in methods.
    #         num_clients = int(num_available_clients * self.fraction_fit)
    #         clients_to_do_fit = max(num_clients, self.min_fit_clients)
    # ```
    strategy = instantiate(
        cfg.strategy.strategy,
        num_classes=cfg.num_classes,
        save_dir=client_progress,
        fraction_fit=cfg.C_fraction,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        # min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        fraction_evaluate=cfg.C_fraction,  # similar to fraction_fit, we don't need to use this argument.
        # min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(cfg),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=evaluate_fn, # a function to run on the server side to evaluate the global model.
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average, 
        )
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=0.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
    #     min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
    #     fraction_evaluate=0.0,  # similar to fraction_fit, we don't need to use this argument.
    #     min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
    #     min_available_clients=cfg.num_clients,  # total clients in the simulation
    #     on_fit_config_fn=get_on_fit_config(cfg),  # a function to execute to obtain the configuration to send to the clients during fit()
    #     evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader), # a function to run on the server side to evaluate the global model.
    #     evaluate_metrics_aggregation_fn=weighted_average,
    #     fit_metrics_aggregation_fn=weighted_average,
    #     )

    # Define server
    server = Server(strategy=strategy, client_manager=SimpleClientManager())
    if isinstance(strategy, ScaffoldStrategy):
        print("Chose SCAFFOLD alg!")
        server= ScaffoldServer(strategy=strategy, num_classes=cfg.num_classes, client_manager=SimpleClientManager())

    ## 5. Start Simulation
    num_cpus = 6
    num_gpus = 1
    ram_memory = 16_000 * 1024 * 1024 # 16 GB
    history = fl.simulation.start_simulation(
        strategy=strategy,  # our strategy of choice
        server=server,
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        # strategy=strategy,  # our strategy of choice
        ray_init_args = {
            "include_dashboard": True}, # we need this one for tracking
    #         "num_cpus": num_cpus,
    #         "num_gpus": num_gpus,
    #         # "memory": ram_memory,
    # },
        client_resources=cfg.client_resources,
        # client_resources=cfg.client_resources,  # (optional) controls the degree of parallelism of your simulation.
        # Lower resources per client allow for more clients to run concurrently
        # (but need to be set taking into account the compute/memory footprint of your run)
        # `num_cpus` is an absolute number (integer) indicating the number of threads a client should be allocated
        # `num_gpus` is a ratio indicating the portion of gpu memory that a client needs.
    )

    ## 6. Save your results
    print("................")
    print(history)
    save_path = HydraConfig.get().runtime.output_dir
    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})

    # results_path = Path(save_path) / "results.pkl"
    # results = {"history": history, "anythingelse": "here"}
    # # save the results as a python pickle
    # with open(str(results_path), "wb") as h:
    #     pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    # Test centralized
    rounds, test_loss = zip(*history.losses_centralized)
    _, test_accuracy = zip(*history.metrics_centralized["accuracy"])
    # Fit metrics
    _, train_loss = zip(*history.metrics_distributed_fit["loss"])
    _, train_acc = zip(*history.metrics_distributed_fit["accuracy"])
    _, mean_diff_acc = zip(*history.metrics_distributed_fit["mean_diff_acc"])
    _, var_diff_acc = zip(*history.metrics_distributed_fit["var_diff_acc"])
    # Evaluation metrics
    _, val_loss = zip(*history.metrics_distributed["loss"])
    _, val_acc = zip(*history.metrics_distributed["accuracy"])

    if len(rounds) != cfg.num_rounds:
        # drop zeroth evaluation round before start of training
        test_loss = test_loss[1:]
        test_accuracy = test_accuracy[1:]
        rounds = rounds[1:]

    file_suffix: str = (
        f"{cfg.strategy.name}"
        f"_{cfg.exp_name}"
        f"{'_varEpoch' if cfg.var_local_epochs else ''}"
        f"_{cfg.partitioning}"
        # f"{'_alpha{cfg.alpha}' if cfg.partitioning == "dirichlet" else ''}"
        f"{'_balanced' if cfg.balance else ''}"
        f"_Classes={cfg.num_classes}"
        f"_Seed={cfg.seed}"
        f"_C={cfg.num_clients}"
        f"_fraction{cfg.C_fraction}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.local_epochs}"
        f"_R={cfg.num_rounds}"
    )
    plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
    )

    file_name = os.path.join(
        save_path,
        f"{file_suffix}.csv",
    )
    df = pd.DataFrame(
        {"round": rounds, "test_loss": test_loss, "test_accuracy": test_accuracy, 
         "train_loss": train_loss, "train_acc": train_acc, "mean_diff_acc": mean_diff_acc, "var_diff_acc": var_diff_acc, "val_loss": val_loss, "val_acc": val_acc}
    )
    df.to_csv(file_name, index=False)
    print(f"---------Experiment Completed in : {(time.time()-start)/60} minutes")

if __name__ == "__main__":
    main()