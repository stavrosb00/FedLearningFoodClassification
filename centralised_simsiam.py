import torch
import torch.nn as nn
from centr_utils import *
# from data_process import *
# from train import *
from model import train, test, ResNet18, train_loop, train_loop_ssl, SimSiam, adjust_learning_rate
from dataset import load_centr_data_SSL
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import time
import argparse
from tqdm import tqdm
from knn_monitor import knn_monitor

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    start = time.time()
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will be on {DEVICE.type}")

    
    #model type architecture
    model = 'simsiam_resnet18'
    num_workers = cfg.num_workers
    bs = cfg.batch_size
    epochs = cfg.num_rounds
    # warm_start = args.warm_start
    warm_start = False
    #subset or full
    # subset = 'subset'
    subset = cfg.subset
    n_classes = cfg.num_classes
    #initialize module
    if subset:
        net = ResNet18(n_classes, pretrained= False)
        print(f"Subset of {n_classes} food categories!")
    else:
        net = ResNet18(101, pretrained= False)
        print("hello22") #subset == 'subset' kai subset=True tote ebaze 101

    net = SimSiam(backbone=net.resnet, hidden_dim=2048, pred_dim=512, output_dim=2048)
    datapath = cfg.datapath  #"D:/DesktopC/Datasets/data/"
    #loading data
    trainloader, testloader, memoryloader = load_centr_data_SSL(datapath=datapath, 
                 subset=subset,
                 num_classes=n_classes,
                 num_workers=num_workers,
                    batch_size=bs)
    print("Data loaded")
    # warm_start = False
    if warm_start == True:
        copy_model(net, model, subset)
    init_lr = cfg.optimizer.lr * bs / 256 # linear scheduling scale
    # parameters grouping change if fix_lr is wanted on encoder or predictor 
    optimizer = torch.optim.SGD(net.parameters(), lr=init_lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    info_dict = train_loop_ssl(net=net, trainloader=trainloader, testloader=testloader, memoryloader=memoryloader, optimizer=optimizer, epochs=epochs, device=DEVICE, init_lr=init_lr)
    # Plot and save results
    save_path = HydraConfig.get().runtime.output_dir
    plot_results_SSL(save_path=save_path, info_dict=info_dict, subset=subset, num_classes=n_classes, model=model)
    # Save the current net module 
    if not os.path.exists(cfg.checkpoint_path):
            os.makedirs(cfg.checkpoint_path)           
    prefix_net: str = f"{cfg.checkpoint_path}centr_pretrained_{model}_classes{n_classes}_E{epochs}.pth"
    torch.save(net.state_dict(), prefix_net)
    print("Results saved, exiting succesfully...")
    print(f"---------Experiment Completed in : {(time.time()-start)/60} minutes")
if __name__ == "__main__":
    # parser
    main()
    pass