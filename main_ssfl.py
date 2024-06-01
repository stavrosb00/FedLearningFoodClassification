import pickle
from pathlib import Path
import time
import os
import pandas as pd
import torch
import numpy as np
from collections import OrderedDict
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from hydra.utils import call, instantiate
from dataset import load_dataset_SSL
from client import generate_client_fn
from client_scaffold import ScaffoldClient
from client_ssfl import generate_client_fn
from strategy import get_on_fit_config_ssfl, get_evaluate_fn_ssfl, weighted_average_ssfl, CustomFedAvgStrategy #get_metrics_aggregation_fn
from utils import save_results_as_pickle, plot_metric_from_history_ssfl
from strategy_scaffold import ScaffoldStrategy, ScaffoldStrategyV2
from strategy_ssfl import HeteroSSFLStrategy, FedSimSiamStrategy
from server_scaffold import ScaffoldServer, ScaffoldServerV2
from server import FedAvgServer, HeteroSSFLServer
from model import ResNet18, test, SimSiam, get_model
from knn_monitor import knn_monitor
import random
import warnings
# Suppress deprecation warnings above tensorboard 
# warnings.filterwarnings("ignore", category=DeprecationWarning)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from torch.utils.tensorboard import SummaryWriter

# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    start = time.time()
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    ## library me centralised pre-trained models se alla BIG FOOD datasets gia meta-learning se clients me mikra datasets
    # model = cfg.model instantiate() call()
    ## 2. Prepare your dataset
    trainloaders, validationloaders, testloader, memoryloader, radloader = load_dataset_SSL(cfg.datapath, cfg.subset, cfg.num_classes, cfg.num_workers, 
                                                        cfg.num_clients, cfg.batch_size, cfg.partitioning, cfg.alpha, cfg.balance, cfg.seed, val_ratio = 0, rad_ratio=cfg.rad_ratio)


    checkpoint_path: str = (
        f"{cfg.checkpoint_path}best_model_eval_"
        f"{cfg.strategy.name}"
        f"_{cfg.exp_name}"
        f"{'_varEpoch' if cfg.var_local_epochs else ''}"
        f"_{cfg.partitioning}"
        f"{'_alpha' + str(cfg.alpha) if cfg.partitioning == 'dirichlet' else ''}"
        f"{'_balanced' if cfg.balance else ''}"
        f"_Classes={cfg.num_classes}"
        f"_Seed={cfg.seed}"
        f"_C={cfg.num_clients}"
        f"_fraction{cfg.C_fraction}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.local_epochs}"
        f"_R={cfg.num_rounds}"
    )
    if cfg.load:
        try:
            checkpoint = np.load(
                f"{checkpoint_path}.npz",
                allow_pickle=True,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model = ResNet18(cfg.num_classes)
            model = SimSiam(backbone=ResNet18(cfg.num_classes, pretrained=False).resnet)
            npz_keys = [key for key in checkpoint.keys() if key.startswith('array')]
            params_dict = zip(model.state_dict().keys(), [checkpoint[key] for key in npz_keys])
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict)
            # If someone wants to choose his testloader related to our artitificla knowledgeable memoryloader.
            acc = knn_monitor(model.encoder.to(device), memoryloader, testloader, k=min(25, len(memoryloader.dataset)), device=device, hide_progress=True)
            # loss, acc = test(model, testloader, device)
            print(f"--kNN Accuracy: {acc:.3f} on Test set, Round: {checkpoint['scalar_2']} --")
            # print(f"-L {checkpoint['scalar_0']} -A {checkpoint['scalar_1']:.3f} - R: {checkpoint['scalar_2']}")
        except FileNotFoundError():
            print(f"Checkpoint path {checkpoint_path} to be loaded is not implemented.")
        return None

    save_path = HydraConfig.get().runtime.output_dir
    log_dir = os.path.join(save_path, "logger")
    print(f"Logger workdir: {log_dir}")
    writer = SummaryWriter(log_dir= log_dir)
    # writer.close()
    #tensorboard --logdir=outputs\2024-05-13\21-45-27 etc. or outputs for all runs 

    # Warm-start option: ...

    ## 3. Define your clients
    if cfg.strategy.client_fn._target_ == "client_ssfl.generate_client_fn":
        client_progress = os.path.join(save_path, "clients")
        print("Local progress for clients who participated in rounds are saved to: ", client_progress)
        # generate_client_fn(trainloaders, validationloaders, radloader, cfg.num_classes, cfg.local_epochs, cfg, save_dir=client_progress)
        client_fn = call(cfg.strategy.client_fn, trainloaders, validationloaders, radloader, cfg.num_classes, cfg.local_epochs, cfg, save_dir=client_progress)
        evaluate_fn = get_evaluate_fn_ssfl(cfg.num_classes, testloader, memoryloader)
        
        mdl = SimSiam(backbone=ResNet18(cfg.num_classes, pretrained=cfg.pretrained).resnet)
        params = [val.cpu().numpy() for _, val in mdl.state_dict().items()]
        L = len(radloader.dataset) #Length of RAD 
        d_phi = mdl.encoder[1].l3[0].out_features # Features dimension length of encoder's output
        params.append(np.random.randn(L, d_phi))
        print(f"NDArrays buffer length {len(params)}")
        params = fl.common.ndarrays_to_parameters(params)
        mdl = None
    elif cfg.strategy.client_fn._target_ == "client_ssfl.generate_client_fedsimsiam_fn":
        client_progress = os.path.join(save_path, "clients")
        print("Local progress for clients who participated in rounds are saved to: ", client_progress)
        client_fn = call(cfg.strategy.client_fn, trainloaders, validationloaders, radloader, cfg.num_classes, cfg.local_epochs, cfg, save_dir=client_progress)
        evaluate_fn = get_evaluate_fn_ssfl(cfg.num_classes, testloader, memoryloader)
        mdl = SimSiam(backbone=ResNet18(cfg.num_classes, pretrained=cfg.pretrained).resnet)
        if cfg.warm_start:
            print(f"Warm start - retrieving FedSimSiam model from {cfg.model.checkpoint}")
            mdl = get_model(model=mdl, pretrained_model_path=cfg.model.checkpoint)
            
        params = [val.cpu().numpy() for _, val in mdl.state_dict().items()]
        print(f"NDArrays buffer length {len(params)}")
        params = fl.common.ndarrays_to_parameters(params)
        mdl = None
    else:
        raise NotImplementedError("SSFL routine not implemented.")
    ## 4. Define your strategy
    # HeteroSSFLStrategy()
    strategy = instantiate(
        cfg.strategy.strategy,
        num_classes=cfg.num_classes,
        checkpoint_path = checkpoint_path,
        writer = writer,
        save_dir=client_progress,
        initial_parameters=params,
        fraction_fit=cfg.C_fraction,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        fraction_evaluate=0.0,  # similar to fraction_fit, we don't need to use this argument.
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config_ssfl(cfg), #cosine decay LR scheduling server side on Round level dependance for client's fit() method
        evaluate_fn=evaluate_fn, # a function to run on the server side to evaluate the global model.
        evaluate_metrics_aggregation_fn=weighted_average_ssfl,
        fit_metrics_aggregation_fn=weighted_average_ssfl, 
        )

    # Define server
    # random.seed = 2024
    # random.seed(cfg.seed) # Reset random seed state clock , Client Manager sampling clients based on random
    if isinstance(strategy, HeteroSSFLStrategy):
        server = HeteroSSFLServer(strategy=strategy, num_classes=cfg.num_classes, checkpoint_path=checkpoint_path, client_manager=SimpleClientManager())
    elif isinstance(strategy, FedSimSiamStrategy): 
        server = Server(strategy=strategy, client_manager=SimpleClientManager())
    else:
        raise NotImplementedError("Strategy not implemented for SSFL settings")
    
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
        # ray_init_args = {
        #     "include_dashboard": True}, # we need this one for tracking
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
    if cfg.cleaner and client_progress != None:
        progress_path = Path(client_progress)
        delete_files = progress_path.glob("client_cv*.pt")
        for f in delete_files:
            f.unlink()
        
        print("[DELETED]Memory cost temp .pt format files related to client states")

    ## 6. Save your results
    # Close summary writer
    writer.close()
    print("................")
    print(history)
    # save_path = HydraConfig.get().runtime.output_dir
    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})
    # Test centralized
    # print(f"{(time.time()-start)/60} minutes")
    # if strategy.evaluate_fn = None:
    # rounds, test_loss = zip(*history.losses_centralized)
    # writer.add_scalar()
    rounds, test_accuracy = zip(*history.metrics_centralized["accuracy"])

    # _, train_acc = zip(*history.metrics_distributed_fit["accuracy"])
    # Evaluation metrics
    # _, val_loss = zip(*history.metrics_distributed["loss"])
    # _, val_acc = zip(*history.metrics_distributed["accuracy"])

    if len(rounds) != cfg.num_rounds:
        # drop zeroth evaluation round before start of training
        # test_loss = test_loss[1:]
        test_accuracy = test_accuracy[1:]
        rounds = rounds[1:]

    # Fit metrics
    _, train_loss = zip(*history.metrics_distributed_fit["loss"])
    try:
        _, d_loss = zip(*history.metrics_distributed_fit["d_loss"])
    except KeyError:
        d_loss = [0 for i in range(rounds[-1])]
    try:
        _, cka_loss = zip(*history.metrics_distributed_fit["cka_loss"])
    except KeyError:
        cka_loss = [0 for i in range(rounds[-1])]

    file_suffix: str = (
        f"{cfg.strategy.name}"
        f"_{cfg.exp_name}"
        f"{'_varEpoch' if cfg.var_local_epochs else ''}"
        f"_{cfg.partitioning}"
        f"{'_alpha' + str(cfg.alpha) if cfg.partitioning == 'dirichlet' else ''}"
        f"{'_balanced' if cfg.balance else ''}"
        f"_Classes={cfg.num_classes}"
        f"_Seed={cfg.seed}"
        f"_C={cfg.num_clients}"
        f"_fraction{cfg.C_fraction}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.local_epochs}"
        f"_R={cfg.num_rounds}"
    )
    plot_metric_from_history_ssfl(
        test_accuracy,
        train_loss,
        save_path,
        (file_suffix),
    )

    file_name = os.path.join(
        save_path,
        f"{file_suffix}.csv",
    )
    if len(test_accuracy) == len(train_loss):
        df = pd.DataFrame(
        {"round": rounds, "test_accuracy": test_accuracy, 
         "train_loss": train_loss, "d_loss": d_loss, "cka_loss": cka_loss})
    else:
        df = pd.DataFrame(
        {"round": rounds, "train_loss": train_loss, "d_loss": d_loss, "cka_loss": cka_loss})
    
    df.to_csv(file_name, index=False)
    print(f"---------Experiment Completed in : {(time.time()-start)/60} minutes")

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', 'profiles/fed_heterossl_fileR=1E=2_Nwork=0')
    # torch.autograd.set_detect_anomaly(mode=True) # for gradscaler and backwards debugging
    # snakeviz .\profiles\fed_heterossl_fileR=1E=1
    main()
    pass