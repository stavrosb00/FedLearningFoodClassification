import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from typing import Dict, List
from utils import comp_accuracy
from torch.optim import SGD
from torch.cuda.amp import autocast, GradScaler

# Note the model and functions here defined do not have any FL-specific components.
class VGG(nn.Module):
    """VGG model."""

    def __init__(self):
        super().__init__()
        self.features = make_layers(cfg["A"])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        """Forward pass through the network."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(network_cfg, batch_norm=False):
    """Define the layer configuration of the VGG-16 network."""
    layers = []
    in_channels = 3
    for v in network_cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class ResNet18(nn.Module):
    """Initialize ResNet18 architecture as module"""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # self.resnet = torchvision.models.resnet18(weights= None)
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        return x

class Net(nn.Module):
    def __init__(self, num_classes: int):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)			#Conv2d(input image channel (3 channels (CIFAR - R G B) or 1 (MNIST - Grayscale), 6 output channels, 5x5 square convolution kernel) -> [28X28x6]
        self.pool = nn.MaxPool2d(2, 2) 			#2x2 maxpooling (kernel_size=2, stride=2) -> [14x14x6]
        self.conv2 = nn.Conv2d(6, 16, 5)		#Conv2d(6 input, 16 output, 5x5 conv kernel) -> [10x10x16]
       	#Linear layers y = Wx + b (fully connected layers to 10 output classes)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)		
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def train(net, trainloader, optimizer, epochs, device: str, proximal_mu = 0):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    # with torch.autocast(device_type="cuda"):
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    if proximal_mu > 0.0:
        global_params = [val.detach().clone() for val in net.parameters()]
    else:
        global_params = None
    net.train()
    train_losses = []
    train_accuracy = []
    for _ in range(epochs):
        for images, labels in trainloader:
            #load data to device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            #forward pass
            outputs = net(images)
            #loss = criterion(outputs, labels)
            if global_params is None:
                loss = criterion(outputs, labels)
            else:
                # Proximal updates for FedProx
                proximal_term = 0.0
                for local_weights, global_weights in zip(
                    net.parameters(), global_params
                ):
                    proximal_term += torch.square(
                        (local_weights - global_weights).norm(2)
                    )
                loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term

            #backward pass
            loss.backward()
            #gradient step
            optimizer.step()

            acc = comp_accuracy(outputs, labels)

            train_losses.append(loss.item())
            train_accuracy.append(acc[0].item())

    train_loss = sum(train_losses) / len(train_losses)
    train_acc = sum(train_accuracy) / len(train_accuracy)

    return train_loss, train_acc



def test(net: nn.Module, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    # with torch.autocast(device_type="cuda"):
    criterion = torch.nn.CrossEntropyLoss()
    accuracy = []
    loss = 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # with autocast():
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            acc1 = comp_accuracy(outputs, labels)
            accuracy.append(acc1[0].item())
    # accuracy = accuracy / len(testloader.dataset)
    return loss / len(testloader.dataset), sum(accuracy) / len(accuracy) 

class ScaffoldOptimizer(SGD):
    """Implements SGD optimizer step function as defined in the SCAFFOLD paper."""

    def __init__(self, params, step_size, momentum, weight_decay):
        super().__init__(
            params=params, lr=step_size, momentum=momentum, weight_decay=weight_decay
        )

    def step_custom(self, server_cv: list[torch.Tensor], client_cv: list[torch.Tensor]):
        """Implement the custom step function for SCAFFOLD (option (ii))."""
        # y_i = y_i - \eta * (g_i + c - c_i)  -->
        # y_i = y_i - \eta*(g_i + \mu*b_{t}) - \eta*(c - c_i)
        # self.step() # y_i = y_i - \eta*(g_i + \mu*b_{t})
        # for group in self.param_groups:
        #     for par, s_cv, c_cv in zip(group["params"], server_cv, client_cv):
        #         par.data.add_(s_cv.to(device='cuda') - c_cv.to(device='cuda'), alpha=-group["lr"]) # y_i' = - \eta*(c - c_i)
        
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cv, client_cv):
                p.data.add_(other=(p.grad.data + sc.to(device='cuda') - cc.to(device='cuda')), alpha=-group['lr'])


        

def train_scaffold(net: nn.Module, trainloader, optimizer: ScaffoldOptimizer, epochs, device, server_cv: list[torch.Tensor], client_cv: list[torch.Tensor]):
    """Train the network based on SCAFFOLD control variates"""
    # with torch.autocast(device_type="cuda"):
    scaler = GradScaler()
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    train_losses = []
    train_accuracy = []
    for _ in range(epochs):
        for images, labels in trainloader:
            #load data to device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            #forward pass
            # with autocast():
            outputs = net(images)
            loss = criterion(outputs, labels)
            #backward pass
            # scaler.scale(loss).backward()
            loss.backward()
            #gradient step for SCAFFOLD class
            # scaler.unscale_(optimizer)
            optimizer.step_custom(server_cv=server_cv, client_cv=client_cv)
            # scaler.update()

            acc = comp_accuracy(outputs, labels)
            train_losses.append(loss.item())
            train_accuracy.append(acc[0].item())

    train_loss = sum(train_losses) / len(train_losses)
    train_acc = sum(train_accuracy) / len(train_accuracy)

    return train_loss, train_acc

#Centralised training loop
def train_loop(net: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          proximal_mu=0) -> Dict[str, List]:
    """Train the network on the training set.
    This is a training loop for Centralised Learning with PyTorch.
    Trains and tests a PyTorch model.

    Passes a target PyTorch models through train() and test()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.
    """
    
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    # criterion =  torch.nn.CrossEntropyLoss()
    net.to(device)
    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc = train(net=net,
                                          trainloader=train_dataloader,
                                          optimizer=optimizer,
                                          epochs=1,
                                          device=device,
                                          proximal_mu=proximal_mu)
        test_loss, test_acc = test(net=net,
          testloader=test_dataloader,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

    # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    print('Finished Training')
    # Return the filled results at the end of the epochs
    return results
    