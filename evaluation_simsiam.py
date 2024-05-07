import torch
import torch.nn as nn
from centr_utils import *
# from data_process import *
# from train import *
from model import train, test, ResNet18, train_loop, train_loop_ssl, SimSiam, adjust_learning_rate
from dataset import load_centr_data
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import time
import argparse
from tqdm import tqdm
from knn_monitor import knn_monitor

class LinearEvaluationSimSiam(nn.Module):
    def __init__(self, pretrained_model_path, device, linear_eval=True, num_classes: int = 10):
        super(LinearEvaluationSimSiam, self).__init__()
        model = SimSiam()
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        self.encoder = model.encoder.to(device)

        if linear_eval:
            # freeze parameters         
            for param in self.encoder.parameters():
                param.requires_grad = False

        #h diwxnw MLP projector head kai bazw ena Linear sto encoder.fc. Sto forward mexri telos tou trainable fc 1 linear?     
        self.classifier = nn.Linear(in_features = self.encoder[1].l3[0].out_features, out_features = num_classes).to(device)

    def forward(self, x):
        return self.classifier(self.encoder(x))

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    start = time.time()
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will be on {DEVICE.type}")
    #model type architecture
    model = 'resnet18'
    num_workers = cfg.num_workers
    bs = cfg.batch_size
    epochs = cfg.num_rounds
    warm_start = False
    #subset or full
    subset = cfg.subset
    n_classes = cfg.num_classes
    #initialize module
    # TODO : model checkpoint on .npz or .pth format re arrange
    if subset == 'subset':
        net = LinearEvaluationSimSiam(pretrained_model_path=cfg.model.checkpoint, device=DEVICE, linear_eval=True, num_classes=n_classes)
    else:
        net = LinearEvaluationSimSiam(pretrained_model_path=cfg.model.checkpoint, device=DEVICE, linear_eval=True, num_classes=101)

    datapath = cfg.datapath  #"D:/DesktopC/Datasets/data/"
    #loading data
    trainloader, testloader= load_centr_data(datapath=datapath, 
                 subset=subset,
                 num_classes=n_classes,
                 num_workers=num_workers,
                    batch_size=bs)
    print("Data loaded")

    # warm_start = False
    if warm_start == True:
        copy_model(net, model, subset)


    # criterion =  torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # epochs = 5
    # training and results
    info_dict = train_loop(net=net, train_dataloader=trainloader, test_dataloader=testloader, 
                        optimizer=optimizer,epochs=epochs, device=DEVICE)
    
    #save results
    save_path = HydraConfig.get().runtime.output_dir

    plot_results(save_path=save_path, info_dict=info_dict, subset=subset, num_classes=n_classes, model=model)
    # Save the current net module 
    # if not os.path.exists('./models'):
    #         os.makedirs('./models') 
    if not os.path.exists(cfg.checkpoint_path):
            os.makedirs(cfg.checkpoint_path) 
            

    prefix_net: str = f"{cfg.checkpoint_path}centr_model_classes{n_classes}_E{epochs}.pth"
    torch.save(net.state_dict(), prefix_net)
    print("Results saved, exiting succesfully...")
    print(f"---------Experiment Completed in : {(time.time()-start)/60} minutes")
if __name__ == "__main__":
    # parser
    main()
    pass