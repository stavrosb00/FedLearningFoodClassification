import copy
import torch
import numpy as np
from torch import Tensor
from model import *
from torchvision import models
from typing import OrderedDict, Type
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out


class ResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block,
        num_classes: int  
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def model_summary(model: nn.Module):
    buffers = []
    parameters = []

    for name, tensor in model.named_buffers():
        buffers.append((name, tensor.clone().detach()))

    for name, param in model.named_parameters():
        parameters.append((name, param.clone().detach()))
    
    # Print buffer names and sizes
    print("Buffers:")
    for name, tensor in buffers:
        print(name, tensor.size())

    # Print parameter names and sizes
    print("\nParameters:")
    for name, param in parameters:
        print(name, param.size())
    
    print(len(buffers))
    print(len(parameters))
    return (buffers, parameters)

def compare_model_parameters(model1, model2):
    params_model1 = model1.state_dict()
    params_model2 = model2.state_dict()

    # Check if the keys are the same
    if params_model1.keys() != params_model2.keys():
        print("Models have different sets of parameters.")
        return False

    # Check if the values are the same
    for key in params_model1.keys():
        if not torch.equal(params_model1[key], params_model2[key]):
            print(f"Parameters for '{key}' are different.")
            return False

    print("Models have the same parameters.")
    return True

def main():
    model = ResNet18(4)
    client_cv = []
    for _, param in model.state_dict().items():
                client_cv.append(param)
                print(client_cv[-1].shape, type(client_cv[-1]), client_cv[-1].device)
    
    return 0
    temp_model = copy.deepcopy(model)
    buffers, parameters = model_summary(model=model)
    print(type(buffers[0][0]))
    print(type(buffers[0][1]))
    buffers = OrderedDict({k: v for k, v in buffers})
    model.load_state_dict(buffers, strict=False)


    same_parameters = compare_model_parameters(model, temp_model)
    if same_parameters:
        print("Models have the same parameters.")
    else:
        print("Models have different parameters.")
    
    count = [p for p in model.parameters()]
    # print(len(count))
    print(len(count))
    l1 = [i for i in range(50)]
    l2 = l1[10 : 2* 10]
    print(l2)
    print(len(l2))
    print(l1)
    print(l1[2*10 :])
    ct_p = 0
    ct_b = 0
    grad_map: list[bool] = [p.requires_grad for _,p in model.state_dict(keep_vars=True).items()]
    for grad, i in zip(grad_map, range(len(grad_map))):
        if grad:
            print(grad, i)
            ct_p+= 1
        else:
            ct_b+= 1

    
    print(ct_p)
    print(ct_b)
    print(np.sqrt(5)[0])
    # ct_p: int = 0
    # ct_b: int = 0
    # for i, (key, p) in enumerate(model.state_dict(keep_vars=True).items()):
    #     if p.requires_grad:
    #         grad_map.append(grad_map[i]) = True
    #         ct_p+=1
    #     else:
    #         grad_map[i] = False
    #         ct_b+=1

    # print(ct_p)
    # print(ct_b)      
    # print(grad_map)
    return 0
    # model.load_state_dict(strict=False, )
    for param in model.parameters():
            client_cv.append(param.clone().detach())
    
    print(len(client_cv))
    print(len(model.state_dict()))
    # for c_i in client_cv:
    #     print(c_i[0][0])
    #     c_i.data = c_i + 0.02
    #     print(c_i[0][0])
    #     break
    # print(client_cv[0][0][0])
    return 0
    # model = ResNet(3, 18, BasicBlock, 4)
    for k, p in model.named_parameters():
        print(k, p.shape)
        
    bufs = [buf for buf in model.buffers()]
    # for k, p in model.state_dict().items():
    #     print(k, p.shape)

    # for b in bufs:
    #     print(b.shape)
    print(len(bufs)) #60
    # for buf in model.buffers():
    #     print(type(buf), buf.size())
    return 0
    state_dict = OrderedDict({k: v.shape for k, v in model.state_dict().items()}) # if ("running_mean" or "running_var" or "num_batches_tracked") not in k})
    # for k in state_dict.keys():
    #     if "running_mean" in k:
    #         del state_dict["running_mean"]
    # params_dict = zip(model.named_parameters()[0], parameters)
    parameters = [val.detach().numpy() for _, val in model.named_parameters()]

    keys = [k for k, _ in model.named_parameters()]
    params_dict = zip(keys, parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)
    print([p.shape for p in parameters])
    return 0
    for (k, p), s_k, s_val in zip(params, state_dict.keys(), state_dict.values()):
        print(k)
        print(p)
        print(s_k)
        print(s_val)
        if p != s_val:
            break
        if s_k == 'resnet.bn1.running_mean':
            print(s_val)
            break

    print(len(state_dict))
    print(len(params))
    return 0
    # model = VGG()
    # model = models.resnet18()
    # print(len(model.parameters()))
    # bn_state = {
    #         name: val.cpu().numpy()
    #         for name, val in model.state_dict().items()
    #         if "bn" in name
    #     }


    client_cv=[]
    for param in model.parameters():
            client_cv.append(torch.zeros_like(param))
    # for param in model.parameters():
    #         # print(type(param), param.size(), param.shape)
    #         client_cv.append(torch.zeros(param.shape))
    # print(len(client_cv))
    print(len(client_cv))
    print(type(client_cv))
    client_cv =[]
    for param in model.parameters():
            client_cv.append(param.clone().detach())
    print(len(client_cv))
    print(type(client_cv))
    print(type((client_cv[0])))
    # print(client_cv[0])
    # print(torch.unsqueeze)      
    a = model.parameters()
    print(a)
    print(len(model.state_dict().keys()))
    y_i = [val.cpu().numpy() for _, val in model.state_dict().items()]
    print([v.shape for _, v in model.state_dict().items()])
    params = [v for _, v in model.state_dict().items()]
    print(len(params))
    num_parameters = sum(p.numel() for p in model.parameters())
    num_state_dict = sum(p.numel() for p in model.state_dict().values())
    print('num parameters = {}, stored in state_dict = {}, diff = {}'.format(num_parameters, num_state_dict, num_state_dict - num_parameters))
    # params2 = [
    #         val["cum_grad"].cpu().numpy()
    #         for _, val in self.optimizer.state_dict()["state"].items()
    #     ]
    #     return params
    # print([val.cpu().numpy().shape for _, val in model.state_dict().items()])
    # print(model.state_dict().items())
    # print(y_i) #122 nd_arrays
    # print(type(y_i))
    # print(type((y_i[0])))
    # print(len(y_i))
    # for i in range(60):
    #     #   print(y_i[i])
    #       print(y_i[i].shape)
    #       print(type(y_i[i]))
    #       print(y_i[i])
    #     #   print(client_cv[i])
    #     #   print(client_cv[i].shape)
    # print(y_i)
    # return 0
    # for group in self.param_groups:
    # server_cv

    i=0
    for p, sc, cc in zip(model.parameters(), params, client_cv):
        i +=1
        # print(type(p))
        # d_p = p.grad.data
        # print(d_p.shape)
        # print(len(p))
        print(p.shape)
        # print(len(cc))
        print(sc.shape)
        print(cc.shape)
        p.data.add_(other=(p.data + sc - cc), alpha=-0.001)
    
    print(i)
    # params = ndarrays_to_parameters(model.parameters())
    # print(type(params))
    # server_cv = [
    #         torch.from_numpy(t)
    #         for t in parameters_to_ndarrays(model.parameters())
    #     ]
    # print(len(server_cv))
    # print(type(server_cv))
    # print(model.state_dict().keys())
    # print(model.resnet)
    # torch.Tensor.add_()


if __name__ == "__main__":
    main()