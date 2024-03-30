import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.tensorboard
import torch.utils.tensorboard.summary
import torchvision
import math
from typing import Dict, List, OrderedDict, Tuple
from utils import comp_accuracy
from torch.optim import SGD
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from knn_monitor import knn_monitor
import numpy as np
from linear_cka import linear_CKA_fast
from lr_scheduler import LR_Scheduler
import time
from sklearn.metrics import confusion_matrix

# Note the model and functions here defined do not have any FL-specific components.
class ResNet18(nn.Module):
    """Initialize ResNet18 architecture as module"""
    def __init__(self, num_classes: int = 4, pretrained = True):
        super().__init__()
        self.transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        if pretrained:
            self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = torchvision.models.resnet18()
        # self.resnet = torchvision.models.resnet18(weights= None)
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)
        
    def forward(self, x):
        # x = self.resnet(x)
        return self.resnet(x)
    
# Siamese Network Composition
class ProjectionMLP(nn.Module):
    """Projection MLP g"""
    def __init__(self, in_features, h1_features, h2_features, out_features):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features, h1_features, bias=False),
            nn.BatchNorm1d(h1_features),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(h1_features, h2_features, bias=False),
            nn.BatchNorm1d(h2_features),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(h1_features, out_features, bias=False),
            nn.BatchNorm1d(out_features)
        )
        #to facebook paper bazei 3 layers me teleutaio Linear->BN. EasyFL leei me sketo linear sto end layer
    def forward(self, x):
        # x = self.l1(x)
        # # x = self.l2(x) # FedSSL 2022 paper suggests 2-layer
        # x = self.l3(x)
        return self.l3(self.l2(self.l1(x)))
        # return self.l3(self.l1(x))

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
    if version == 'simplified':# same thing, much faster
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    elif version == 'original':
        z = z.detach() # stop gradient operant
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()
    else:
        raise Exception

class SimSiam(nn.Module):
    def __init__(self, backbone=ResNet18(num_classes=10).resnet, hidden_dim = 2048, pred_dim = 512, output_dim = 2048):
        super(SimSiam, self).__init__()
        # backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity() # erase last linear layer. removal doesn't offer much MB advantage 
        # self.backbone = backbone # comment , parameters economy.
        # 14 array sets of params saved
        # self.projector = ProjectionMLP(backbone.layer4[-1].conv2.out_channels, hidden_dim, hidden_dim, hidden_dim)   #<-buffer econ, comment for after 15 May feature for model chekpt

        # backbone.fc = ProjectionMLP(backbone.layer4[-1].conv2.out_channels, hidden_dim, hidden_dim, hidden_dim) 
        # self.encoder = backbone
        self.encoder = nn.Sequential(
            backbone,
            ProjectionMLP(backbone.layer4[-1].conv2.out_channels, hidden_dim, hidden_dim, hidden_dim)
        )
        self.predictor = PredictionMLP(hidden_dim, pred_dim, output_dim)

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2 # symmetric x0.5
        # return {'loss': L}
        return L

def get_model(model, pretrained_model_path):
    if pretrained_model_path.endswith('.npz'):
        # return "npz"
        checkpoint = np.load(
                pretrained_model_path,
                allow_pickle=True,
            )
        npz_keys = [key for key in checkpoint.keys() if key.startswith('array')]
        try: 
            params_dict = zip(model.state_dict().keys(), [checkpoint[key] for key in npz_keys])
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            # print(len(state_dict))
            model.load_state_dict(state_dict)
            return model
        except:
            nd_arrays_shapes = [checkpoint[key].shape for key in npz_keys]
            print(f"NPZ length of param dict: {len(nd_arrays_shapes)}")
            print(f"nn.Module length of state dict: {len(model.state_dict().keys())}")
            raise ValueError("Miss matched array params")
    elif pretrained_model_path.endswith('.pth'):
        try:
            model.load_state_dict(torch.load(pretrained_model_path))#, map_location=device))
            return model
        except:
            raise ValueError("Miss matched model format params")
    else:
        # pretrained_model_path = 'models/centr_pretrained_0.012_simsiam_resnet18_classes10_E200.pth'
        # pretrained_model_path = "models/best_model_eval_heterossfl_heterossfl_dirichlet_alpha0.5_balanced_Classes=10_Seed=2024_C=5_fraction1_B=128_E=20_R=10.npz"
        raise ValueError("Unsupported file format for model checkpoint. Only .npz and .pth are supported.")
    
class LinearEvaluationSimSiam(nn.Module):
    def __init__(self, model: SimSiam, device, linear_eval=True, num_classes: int = 10): # backbone,
        super(LinearEvaluationSimSiam, self).__init__()
        self.encoder = model.encoder.to(device)
        # freeze parameters 
        if linear_eval:
            self.encoder.requires_grad_(False)        
        #h diwxnw MLP projector head kai bazw ena Linear sto encoder.fc. Sto forward mexri telos tou trainable fc 1 linear?     
        self.classifier = nn.Linear(in_features = self.encoder[1].l3[0].out_features, out_features = num_classes).to(device)

    def forward(self, x):
        return self.classifier(self.encoder(x))

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
    if isinstance(net, LinearEvaluationSimSiam):
        # print("net has encoder")
        net.encoder.eval()
        net.classifier.train()
    else:  
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

#Centralised training loops
def train_loop(net: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          proximal_mu=0) -> Tuple[Dict[str, List], np.ndarray]:
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
    cm: np.ndarray = None
    # criterion =  torch.nn.CrossEntropyLoss()
    net.to(device)
    # Loop through training and testing steps for a number of epochs
    global_progress = tqdm(range(epochs), desc=f'Training')
    for epoch in global_progress:
        criterion = torch.nn.CrossEntropyLoss()
        net.to(device)
        if proximal_mu > 0.0:
            global_params = [val.detach().clone() for val in net.parameters()]
        else:
            global_params = None
        if isinstance(net, LinearEvaluationSimSiam):
            # print("net has encoder")
            net.encoder.eval()
            net.classifier.train()
        else:  
            net.train()
        train_losses = []
        train_accuracy = []
        for images, labels in train_dataloader:
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
        # Using test_loader set per epoch 
        accuracy = []
        loss = 0.0
        net.eval()
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = net(images)
                    loss += criterion(outputs, labels).item()
                acc1 = comp_accuracy(outputs, labels)
                accuracy.append(acc1[0].item())
                if epoch == (epochs-1):
                    all_labels.extend(labels.cpu().numpy())
                    all_outputs.extend(torch.argmax(outputs, 1).cpu().numpy())

        test_loss = loss / len(test_dataloader.dataset)
        test_acc = sum(accuracy) / len(accuracy)
        if epoch == (epochs-1):
            cm = confusion_matrix(all_labels, all_outputs) 
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        # Print out what's happening
        global_progress.set_postfix({
        "epoch":epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc
        })
    print('Finished Training')
    # Return the filled results and confusion matrix at the end of the epochs
    return results, cm
    
def adjust_learning_rate(optimizer: SGD, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs)) # math faster than numpy on single values that aren't arrays
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

# code from https://github.com/PatrickHua/SimSiam/blob/75a7c51362c30e8628ad83949055ef73829ce786/optimizers/__init__.py
def get_optimizer(model: nn.Module, lr: float, momentum: float, weight_decay: float, name: str='sgd'):

    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]
   
    if name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
   
    return optimizer

def train_loop_ssl(net: torch.nn.Module, 
          trainloader: torch.utils.data.DataLoader, 
          testloader: torch.utils.data.DataLoader, 
          memoryloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          init_lr: float,
          lr_scheduler: LR_Scheduler,
          writer: torch.utils.tensorboard.SummaryWriter) -> Dict[str, List]:
    accuracy = 0 
    results = {"knn_accuracy": [],
               "train_loss": []
               }
    # Start training
    net.to(device)
    global_progress = tqdm(range(epochs), desc=f'Training')
    for epoch in global_progress:
        batch_loss = []
        net.train() #need to reset because knn_monitor sets encoder sub-module on eval() mode
        local_progress = tqdm(trainloader, desc=f'Epoch {epoch}/{epochs}')
        for i, data in enumerate(local_progress):
            images = data[0]
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = net(images[0].to(device, non_blocking=True), images[1].to(device, non_blocking=True))
            # loss = data_dict['loss'].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # lr_scheduler.step()
            batch_loss.append(loss)
            local_progress.set_postfix({'loss': loss})

        # accuracy = knn_monitor(net.encoder, testloader, testloader, k=min(25, len(memoryloader.dataset)), device=device)
        accuracy = knn_monitor(net.encoder, memoryloader, testloader, k=min(25, len(memoryloader.dataset)), device=device)  # 25, 7500
        epoch_bs_loss = float(sum(batch_loss) / len(batch_loss))
        writer.add_scalar(f"metrics/kNN_acc", accuracy, epoch)
        info_dict = {"epoch":epoch, "accuracy":accuracy}
        # adjust_learning_rate(optimizer, init_lr, epoch, epochs*2)
        writer.add_scalar(f"metrics/loss", epoch_bs_loss, epoch)
        global_progress.set_postfix(info_dict)
        results["knn_accuracy"].append(float(accuracy))
        results["train_loss"].append(epoch_bs_loss)

    print('Finished Training')
    return results
def train_heterossfl(net: torch.nn.Module, 
          trainloader: torch.utils.data.DataLoader, 
          radloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          mu_coeff: float,
          phi_K_mean: np.ndarray | None,
          hide_progress: bool = False) -> tuple[float, np.ndarray]:

    # Start training
    net.to(device)
    global_progress = tqdm(range(epochs), desc=f'Local training', disable=hide_progress)
    for epoch in global_progress:
        batch_loss = []
        disim_loss = []
        cka_loss = []
        net.train() #need to reset because knn_monitor sets encoder sub-module on eval() mode
        local_progress = tqdm(trainloader, desc=f'Epoch {epoch}/{epochs}', disable=hide_progress)
        # start1= time.time() 
        # vlepe https://github.com/pytorch/examples/blob/6f62fcd361868406a95d71af9349f9a3d03a2c52/imagenet/main.py#L275 
        # me progress meters
        for i, data in enumerate(local_progress):
            # print(f"tr Datal{time.time() - start1}")
            images = data[0]
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = net(images[0].to(device, non_blocking=True), images[1].to(device, non_blocking=True))
                # loss = loss.mean() # minimizing statistical expectation
            # loss = data_dict['loss'].mean()
            if phi_K_mean is None: # first round skip loss assignment
                total_loss = loss
                loss_cka = 0
                # print('First round CKA=0')
                # Glytwnw 1o fresh round me 2 for loops ston radloader otan exw None kai otan den exw None
                if epoch == (epochs-1):
                    embeddings = []
                    with torch.no_grad():
                        for idx, (X_img , _) in enumerate(radloader):
                            # images = data[0] # images[0]
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                embedding = net.encoder(X_img.cuda(non_blocking=True)) # +1000MB sthn gpu logw bs=64. Ana bs=32 ~ 500MB sthn GPU 
                            embeddings.append(embedding)
                        phi_K = torch.cat(embeddings, dim=0).cpu().numpy()
            else:
                embeddings = []
                with torch.no_grad():
                    for idx, (X_img , _) in enumerate(radloader):
                        # images = data[0] # images[0]
                        embedding = net.encoder(X_img.cuda(non_blocking=True)) # +1000MB sthn gpu logw bs=64. Ana bs=32 ~ 500MB sthn GPU 
                        embeddings.append(embedding)
                    phi_K = torch.cat(embeddings, dim=0).cpu().numpy() # [ batch_size x features_dims | batch_size x features_dims | ... | (L-batch_size) x features_dims ]
                loss_cka = mu_coeff * linear_CKA_fast(phi_K, phi_K_mean)
                total_loss = loss - loss_cka # isws kalytera me (-)
            # per local epoch backwards
            total_loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            batch_loss.append(total_loss.item())
            disim_loss.append(loss.item())
            cka_loss.append(loss_cka)
            # local_progress.set_postfix({'loss': float(loss), 'loss_cka': loss_cka, 'total_loss': batch_loss[-1]})
            local_progress.set_postfix({'loss': disim_loss[-1], 'loss_cka': cka_loss[-1], 'total_loss': batch_loss[-1]})
            # start1 = time.time()

        # adjust_learning_rate(optimizer, init_lr, epoch, epochs)
        train_loss = float(sum(batch_loss) / len(batch_loss))
        disim_loss = float(sum(disim_loss) / len(disim_loss))
        cka_loss = float(sum(cka_loss) / len(cka_loss))
        info_dict = {"epoch":epoch, "train_loss":train_loss}
        global_progress.set_postfix(info_dict)
    return train_loss, disim_loss, cka_loss, phi_K

def train_fedsimsiam(net: torch.nn.Module, 
          trainloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          hide_progress: bool = False) -> tuple[float, np.ndarray]:

    # Start training
    net.to(device)
    global_progress = tqdm(range(epochs), desc=f'Local training', disable=hide_progress)
    for epoch in global_progress:
        disim_loss = []
        net.train() #need to reset because knn_monitor sets encoder sub-module on eval() mode
        local_progress = tqdm(trainloader, desc=f'Epoch {epoch}/{epochs}', disable=hide_progress)
        # start1= time.time() 
        # vlepe https://github.com/pytorch/examples/blob/6f62fcd361868406a95d71af9349f9a3d03a2c52/imagenet/main.py#L275 
        # me progress meters
        for i, data in enumerate(local_progress):
            # print(f"tr Datal{time.time() - start1}")
            images = data[0]
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = net(images[0].to(device, non_blocking=True), images[1].to(device, non_blocking=True))
                # loss = loss.mean() # minimizing statistical expectation. Already on forward pass.
            # per local epoch backwards
            loss.backward()
            optimizer.step()
            disim_loss.append(loss.item())
            local_progress.set_postfix({'loss': disim_loss[-1]})
            # start1 = time.time()

        # adjust_learning_rate(optimizer, init_lr, epoch, epochs)
        disim_loss = float(sum(disim_loss) / len(disim_loss))
        info_dict = {"epoch":epoch, "train_loss":disim_loss}
        global_progress.set_postfix(info_dict)
    return disim_loss

if __name__ == "__main__":
    print("Testing things")
    pass
    