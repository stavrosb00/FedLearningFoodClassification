import torch
import math
import time
from model import SimSiam, LinearEvaluationSimSiam
from typing import OrderedDict
import numpy as np

def get_checkpoint_format(pretrained_model_path):
    # Check the file extension
    if pretrained_model_path.endswith('.npz'):
        return "npz"
    elif pretrained_model_path.endswith('.pth'):
        return "pth"
    else:
        raise ValueError("Unsupported file format. Only .npz and .pth are supported.")


def get_model(model, pretrained_model_path, device):
    if pretrained_model_path.endswith('.npz'):
        # return "npz"
        checkpoint = np.load(
                pretrained_model_path,
                allow_pickle=True,
            )
        print(checkpoint)
        npz_keys = [key for key in checkpoint.keys() if key.startswith('array')]
        # nd_arrays_shapes = [checkpoint[key].shape for key in npz_keys]
        # print(len(nd_arrays_shapes))
        # print(len(model.state_dict().keys()))
        params_dict = zip(model.state_dict().keys(), [checkpoint[key] for key in npz_keys])
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        # print(len(state_dict))
        model.load_state_dict(state_dict)
        return model
    elif pretrained_model_path.endswith('.pth'):
        model.load_state_dict(torch.load(pretrained_model_path))#, map_location=device))
        return model
    else:
        raise ValueError("Unsupported file format for model checkpoint. Only .npz and .pth are supported.")
    
def main():
    pretrained_model_path = "models/best_model_eval_heterossfl_heterossfl_dirichlet_alpha0.5_balanced_Classes=10_Seed=2024_C=5_fraction1_B=128_E=20_R=10.npz"
    device = torch.device("cuda")
    model = SimSiam() #backbone=ResNet18().resnet, hidden_dim=2048, pred_dim=512, output_dim=2048
    # model = model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    device = torch.device("cuda")
    model = get_model(model, pretrained_model_path, device)
    print(model)
    linear_mdl = LinearEvaluationSimSiam(model, device, linear_eval=True, num_classes=10)
    print(len(linear_mdl.state_dict()))
    grad_map: list[bool] = [p.requires_grad for _,p in linear_mdl.state_dict(keep_vars=True).items()]
    print(grad_map)

if __name__=='__main__':
    main()
    pass