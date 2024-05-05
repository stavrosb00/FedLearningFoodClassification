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
    
# Siamese Network Composition
class ProjectionMLP(nn.Module):
    """Projection MLP g"""
    def __init__(self, in_features, h1_features, h2_features, out_features):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features, h1_features),
            nn.BatchNorm1d(h1_features),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(h1_features, h2_features),
            nn.BatchNorm1d(h2_features),
            nn.ReLU(inplace=True)

        )
        self.l3 = nn.Sequential(
            nn.Linear(h1_features, out_features),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x):
        # x = self.l1(x)
        # # x = self.l2(x) # FedSSL 2022 paper suggests 2-layer
        # x = self.l3(x)
        return self.l3(self.l1(x))

class PredictionMLP(nn.Module):
    """Prediction MLP h"""
    def __init__(self, in_features, hidden_features, out_features):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        return self.l2(self.l1(x))

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class SimSiam(nn.Module):
    def __init__(self, backbone=ResNet18(num_classes=10), hidden_dim = 2048, pred_dim = 512, output_dim = 2048):
        super(SimSiam, self).__init__()
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity() # erase last linear layer

        self.backbone = backbone
        self.projector = ProjectionMLP(backbone.output_dim, hidden_dim, hidden_dim, hidden_dim)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.predictor = PredictionMLP(hidden_dim, pred_dim, output_dim)

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        # return {'loss': L}
        return L


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

            with torch.autocast(device_type="cuda", dtype=torch.float16):
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
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
    # scaler = GradScaler()
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
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
    