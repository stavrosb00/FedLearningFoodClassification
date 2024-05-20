"""FedAvg Client"""
from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
import torch
import flwr as fl
from hydra.utils import instantiate
from model import SimSiam, ResNet18, train_heterossfl, test, adjust_learning_rate
import numpy as np
import os
import time
import pandas as pd


class HeteroSSFLClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, cid, trainloader, valloader, radloader, num_classes, epochs, config: DictConfig, save_dir) -> None:
        super().__init__()
        self.cid = cid
        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = valloader
        self.radloader = radloader
        # a model that is randomly initialised at first
        self.model = SimSiam(backbone=ResNet18(num_classes, pretrained=False).resnet)#ResNet18(num_classes)
        # SimSiam(backbone=ResNet18(num_classes, pretrained=False).resnet)
        self.len_params = len(self.model.state_dict())
        # self.L = len(self.radloader.dataset) #Length of RAD 
        # self.d_phi = self.model.encoder[1].l3[0].out_features # Features dimension length of encoder's output
        self.phi_K_mean = None
        # figure out if this client has access to GPU support or not
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")
        # print(self.device)
        self.epochs = epochs
        self.exp_config = config
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        # self.var_local_epochs = var_local_epochs

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    
    def fit(self, parameters, config: Dict[str, Scalar]):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """
        start = time.time()
        # phi_K_mean = parameters[self.len_params :]
        # if len(parameters) == self.len_params:
        #     self.phi_K_mean = None
        # else:
        # self.phi_K_mean = parameters[self.len_params :][0]
        self.phi_K_mean = parameters[-1]
        # print(self.phi_K_mean.shape, type(self.phi_K_mean))
        # parameters = parameters[: self.len_params]
        mdl_parameters = parameters[: -1]
        # (print(len(mdl_parameters)))
        # copy parameters sent by the server into client's local model
        self.set_parameters(mdl_parameters)

        # maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        lr = config["lr"] #cosine decay LR scheduling serverside on Round level
        server_round = config["server_round"]
        # num_rounds = config["num_rounds"]
        if server_round == 1:
            self.phi_K_mean = None

        momentum = self.exp_config.optimizer.momentum
        # mu = self.exp_config.optimizer.mu
        weight_decay = self.exp_config.optimizer.weight_decay
        mu_coeff = self.exp_config.optimizer.mu_coeff
        # client's local epochs to train(can be constant or variable choice based on Uniform[var_min_epochs, var_max_epochs])
        if self.exp_config.var_local_epochs:
            seed_val = (
                2024
                + int(self.client_id)
                + int(server_round)
                + int(self.exp_config.seed)
            )
            np.random.seed(seed_val)
            epochs = np.random.randint(
                self.exp_config.var_min_epochs, self.exp_config.var_max_epochs
            )
        else:
            epochs = self.epochs
        # optimizer
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # lr = adjust_learning_rate(optim, lr, server_round, num_rounds)
        # print(f"Client {self.cid} training on {self.device}")
        # local training
        train_loss, d_loss, cka_loss, phi_K = train_heterossfl(self.model, self.trainloader, self.radloader, optim, epochs, self.device, mu_coeff=mu_coeff, phi_K_mean=self.phi_K_mean, hide_progress=True)
        # Possible artificial knowledge for testing: knn_monitor, local memory labeled dataset -> weighted_accuracies strategy side
        # self.get_parameters({}).extend()
        combined_updates = self.get_parameters({}) + [phi_K]
        # return updated model params + phi_K(where batch_size x features), number of examples and dict of metrics
        return combined_updates, len(self.trainloader.dataset), {"loss": train_loss, "d_loss": d_loss, "cka_loss": cka_loss, "client_id" : self.cid, "fit_mins": (time.time()-start)/60}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # return None
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"loss": loss, "accuracy": accuracy, "client_id" : self.cid}


def generate_client_fn(trainloaders, valloaders, radloader, num_classes, epochs, exp_config, save_dir):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        
        # print(f'client id: {cid}')
        return HeteroSSFLClient(
            cid=cid,
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            radloader = radloader,
            num_classes=num_classes,
            epochs=epochs, 
            config=exp_config,
            save_dir=save_dir
        ).to_client()

    # return the function to spawn client
    return client_fn 