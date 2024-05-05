import torch
import torch.nn as nn
from centr_utils import *
# from data_process import *
# from train import *
from model import train, test, ResNet18, train_loop, SimSiam
from dataset import load_centr_data
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import time
import argparse
from knn_monitor import knn_monitor

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
    # warm_start = args.warm_start
    warm_start = False
    #subset or full
    # subset = 'subset'
    subset = cfg.subset
    n_classes = cfg.num_classes
    #initialize module
    if subset == 'subset':
        net = ResNet18(n_classes)
    else:
        net = ResNet18(101)

    datapath = cfg.datapath  #"D:/DesktopC/Datasets/data/"
    imagenet_stats = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    imgenet_mean = imagenet_stats[0]
    imgenet_std = imagenet_stats[1]
    trf = net.transform
    # trf = torchvision.transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(), 
    #         transforms.ToTensor(),
    #         transforms.Normalize(imgenet_mean, imgenet_std)])

    #loading data
    trainloader, testloader= load_centr_data(datapath=datapath, 
                 subset=subset,
                 num_classes=n_classes,
                 num_workers=num_workers,
                    batch_size=bs)
    print("Data loaded")

    show_batch(trainloader, imgenet_std, imgenet_mean, bs)

    # warm_start = False
    if warm_start == True:
        copy_model(net, model, subset)


    # criterion =  torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    # epochs = 5
    # training and results
    info_dict = train_loop(net=net,train_dataloader=trainloader, test_dataloader=testloader, 
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