import torch
import torch.nn as nn
from centr_utils import *
# from data_process import *
# from train import *
from model import train, test, ResNet18, train_loop, train_loop_ssl, SimSiam, adjust_learning_rate, LinearEvaluationSimSiam
from dataset import load_centr_data
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import time
import argparse
from tqdm import tqdm
from knn_monitor import knn_monitor

# Downstream training - linear evaluation : python .\evaluation_linear.py num_rounds=50 batch_size=256 val_ratio=0 optimizer=adam num_workers=0
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
    bs = cfg.batch_size # 64 * 2 = 128
    epochs = cfg.num_rounds # 50 for eval
    lr = cfg.optimizer.lr
    # momentum = cfg.optimizer.momentum
    warm_start = False
    #subset or full
    subset = cfg.subset
    n_classes = cfg.num_classes
    #initialize module
    # TODO : model checkpoint on .npz or .pth format re arrange
    # na to kanw 
    if subset:
        net = LinearEvaluationSimSiam(pretrained_model_path=cfg.model.checkpoint, device=DEVICE, linear_eval=True, num_classes=n_classes)
    else:
        net = LinearEvaluationSimSiam(pretrained_model_path=cfg.model.checkpoint, device=DEVICE, linear_eval=True, num_classes=101)

    # model =f"{model}_downstream_SSL_{cfg.model.checkpoint}"
    # model =f"{model}_downstream_SSL_HeteroSSL"
    model =f"{model}_downstream_SSL_CentralizedV2" # EXP NAME 
    grad_map: list[bool] = [p.requires_grad for _,p in net.state_dict(keep_vars=True).items()]
    print(grad_map)
    print(net)
    # return 0
    datapath = cfg.datapath  #"D:/DesktopC/Datasets/data/"
    # Isws 256x256 kalytera gia downstream
    H = 256
    W = H
    #loading data
    trainloader, testloader= load_centr_data(datapath=datapath, 
                 subset=subset,
                 num_classes=n_classes,
                 num_workers=num_workers,
                    batch_size=bs, H=H, W=W)
    print("Data loaded")

    # warm_start = False
    if warm_start == True:
        copy_model(net, model, subset)


    # criterion =  torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # epochs = 5
    # training and results
    info_dict = train_loop(net=net, train_dataloader=trainloader, test_dataloader=testloader, 
                        optimizer=optimizer,epochs=epochs, device=DEVICE)
    
    #save results
    save_path = HydraConfig.get().runtime.output_dir

    plot_results_downstream(save_path=save_path, info_dict=info_dict, subset=subset, num_classes=n_classes, model=model)
    # plot_results(save_path=save_path, info_dict=info_dict, subset=subset, num_classes=n_classes, model=model)
    # Save the current net module 
    if not os.path.exists(cfg.checkpoint_path):
            os.makedirs(cfg.checkpoint_path) 
            

    prefix_net: str = f"{cfg.checkpoint_path}centr_downstreamed_model_classes{n_classes}_E{epochs}.pth"
    torch.save(net.state_dict(), prefix_net)
    print("Results saved, exiting succesfully...")
    print(f"---------Experiment Completed in : {(time.time()-start)/60} minutes")
if __name__ == "__main__":
    # parser
    main()
    pass