"""SCAFFOLD Client"""
from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
import torch
import flwr as fl
from hydra.utils import instantiate
from model import ResNet18, train, test, ScaffoldOptimizer, train_scaffold
import numpy as np
import os
import time
import pandas as pd
import copy


class ScaffoldClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, cid, trainloader, valloader, num_classes, epochs, config: DictConfig, save_dir: str) -> None:
        super().__init__()
        self.cid = cid
        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = valloader

        # a model that is randomly initialised at first
        # self.model = Net(num_classes)
        self.model = ResNet18(num_classes)
        self.server_cv_model = None
        self.server_cv = None
        self.len_params = len(self.model.state_dict())
        # initialize client control variate with 0 and shape of the module parameters
        self.client_cv: list[torch.Tensor] = []
        for param in self.model.parameters():
            self.client_cv.append(torch.zeros_like(param))

        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.exp_config = config
        # self.var_local_epochs = var_local_epochs

        # self.client_cv = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        

        # for param in self.model.parameters():
        #     self.client_cv.append(torch.zeros(param.shape))
            # torch.zeros_like()
        # save cv to directory
        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def set_model_parameters(self, model, parameters: NDArrays):
        """Receive parameters and apply them to the model."""
        keys = [k for k, _ in model.named_parameters()]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)

        # params_dict = zip(model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        # model.load_state_dict(state_dict, strict=True)

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        # keys = [k for k, _ in self.model.named_parameters()]
        # params_dict = zip(keys, parameters)
        # state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        # self.model.load_state_dict(state_dict, strict=False)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        # return [val.detach().cpu().numpy() for _, val in self.model.named_parameters()]
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Train model received by the server (parameters) using the data
        that belongs to this SCAFFOLD client. Then, send it back to the server.
        """
        start = time.time()
        # the first half are model parameters and the second are the server_cv
        # server_cv = parameters[len(parameters) // 2 :]
        # parameters = parameters[: len(parameters) // 2]
        server_cv = parameters[self.len_params :]
        parameters = parameters[: self.len_params]
        # print(len(server_cv))
        # print(len(parameters))
        # copy model parameters sent by the server into client's local model
        self.set_parameters(parameters)
        # self.server_cv_model = copy.deepcopy(self.model)
        # self.set_model_parameters(self.server_cv_model, server_cv)

        #init client control variate List
        self.client_cv = []
        # load client control variate if not the first fit
        if os.path.exists(f"{self.dir}/client_cv_{self.cid}.pt"):
            self.client_cv = torch.load(f"{self.dir}/client_cv_{self.cid}.pt")
        else: 
            for param in self.model.parameters():
                self.client_cv.append(param.clone().detach())

        # convert the server control variate NDArrays to a list of tensors
        server_cv = [torch.Tensor(cv) for cv in server_cv]
        # server_cv =[param.clone().detach() for param in server_model.parameters()]
        lr = config["lr"]
        server_round = config["server_round"]
        momentum = self.exp_config.optimizer.momentum
        # mu = self.exp_config.optimizer.mu
        weight_decay = self.exp_config.optimizer.weight_decay

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
        # SCAFFOLD optimizer
        optim = ScaffoldOptimizer(self.model.parameters(), lr, momentum, weight_decay)
        # global server's model parameters
        server_model = [p.detach().cpu() for p in self.model.parameters()]
        # server_model = [torch.Tensor(p) for p in parameters]
        # server_model = [torch.from_numpy(p) for p in self.model.parameters()] #copy.deepcopy(self.model.parameters())
        # print(f"Client {self.cid} training on {self.device}")
        # local training
        train_loss, train_acc = train_scaffold(self.model, self.trainloader, optim, epochs, self.device, server_cv, self.client_cv)
        # print(f"training done on {self.cid}")
        # local client's model parameters
        y_i = [p.detach().cpu() for p in self.model.parameters()]
        #self.model.parameters()
        # y_i = [torch.from_numpy(p) for p in self.get_parameters(config={})]

        # self.client_cv = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        # c_i_n = []
        server_update_x = []
        server_update_c = []
        # server_cv ~ self.server_cv_model.parameters()
        # update client control variate c_i' = c_i - c + 1/eta*K*num_batches (x - y_i)
        # j = 0
        for c_i, c, x, yi in zip(self.client_cv, server_cv, server_model, y_i):
            # if j == 0:
            #     print(f"c_i {c_i.device} c {c.device} x {x.device} yi {yi.device}")
            #     j+=1
            
            # c_i' - c_i
            server_update_c.append((- c + (1.0 / (lr * self.epochs * len(self.trainloader))) * (x - yi)).numpy())

            c_i.data = c_i + server_update_c[-1]
            # y_i - x 
            server_update_x.append((yi - x).numpy())

        torch.save(self.client_cv, f"{self.dir}/client_cv_{self.cid}.pt")
        
        # keys = [k for k, _ in self.model.named_parameters()]
        # params_dict = zip(keys, server_update_x)
        # state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        # self.model.load_state_dict(state_dict, strict=False)
        # # concatenate lists (Dy_i, Dc_i)
        # # combined_updates = server_update_x + server_update_c
        # combined_updates = self.get_parameters(config={}) + server_update_c  #+ self.len_params
        # # back to normal trainable param states
        # params_dict = zip(keys, y_i)
        # state_dict = OrderedDict({k: v for k, v in params_dict})
        # self.model.load_state_dict(state_dict, strict=False)
        #p.clone()
        buffers = [p.detach().cpu().numpy() for p in self.model.buffers()]
        # print(f"{self.cid} server update x: {len(server_update_x)} | server update c: {len(server_update_c)} | buffers: {len(buffers)}")
        combined_updates = server_update_x + server_update_c + buffers
        # self.set_parameters(y_i)
        # return combined updates(line13 alg), number of examples and dict of metrics
        return combined_updates, len(self.trainloader.dataset), {"loss": train_loss, "accuracy": train_acc, 
                                                                "client_id" : self.cid, "fit_mins": (time.time()-start)/60}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader.dataset), {"loss": loss, "accuracy": accuracy, "client_id" : self.cid}


def generate_client_fn(trainloaders, valloaders, num_classes, epochs, exp_config, save_dir):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a SCAFFOLDClient with client id `cid`.
    """

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())
        
        # print(f'scaff client id: {cid}')
        return ScaffoldClient(
            cid=cid,
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            num_classes=num_classes,
            epochs=epochs, 
            config=exp_config,
            save_dir=save_dir
        ).to_client()

    # return the function to spawn client
    return client_fn