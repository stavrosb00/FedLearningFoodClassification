"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""




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