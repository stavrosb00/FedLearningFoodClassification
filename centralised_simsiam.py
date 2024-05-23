import torch
import torch.nn as nn
from centr_utils import *
# from data_process import *
# from train import *
from model import train, test, ResNet18, train_loop, train_loop_ssl, SimSiam, adjust_learning_rate, get_optimizer
from dataset import load_centr_data_SSL
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import time
import argparse
from tqdm import tqdm
from knn_monitor import knn_monitor
from lr_scheduler import LR_Scheduler, plot_lr_scheduler
from torch.utils.tensorboard import SummaryWriter

#python .\centralised_simsiam.py num_rounds=50 batch_size=128 val_ratio=0 optimizer=adam num_workers=2
# python .\centralised_simsiam.py num_rounds=200 batch_size=64 val_ratio=0 optimizer=simsiam num_workers=2
# python .\centralised_simsiam.py num_rounds=200 batch_size=128 val_ratio=0 optimizer=simsiam num_workers=2 optimizer.lr=0.032 
# 14/5 stis 10 wra konta: python .\centralised_simsiam.py num_rounds=200 batch_size=128 val_ratio=0 optimizer=simsiam num_workers=2 optimizer.lr=0.05
# Etreksa stis 15/5 apo wra 12-18-53 gia 11.5 hrs: python .\centralised_simsiam.py num_rounds=200 batch_size=128 val_ratio=0 optimizer=simsiam num_workers=0 optimizer.lr=0.05
# ki alla ...
# 20/5 : dokimes me base lr=0.05 python .\centralised_simsiam.py num_rounds=200 batch_size=128 val_ratio=0 optimizer=simsiam num_workers=2 optimizer.lr=0.05
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    start = time.time()
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will be on {DEVICE.type}")

    
    #model type architecture
    model = 'simsiam_resnet18_Jit0.3_min_scale0.4'
    num_workers = cfg.num_workers
    bs = cfg.batch_size
    epochs = cfg.num_rounds
    # warm_start = args.warm_start
    warm_start = False
    #subset or full
    # subset = 'subset'
    subset = cfg.subset
    n_classes = cfg.num_classes
    pretrained = False # Isws pretrained=False kalytera gia sygkrisimothta ws koinh arxh kai oxi unstables basei adaptation phases
    #initialize module
    if subset:
        net = ResNet18(n_classes, pretrained= pretrained) 
        print(f"Subset of {n_classes} food categories!")
    else:
        net = ResNet18(101, pretrained= pretrained)
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
    
    base_lr = cfg.optimizer.lr
    init_lr = base_lr * bs / 256 # linear scheduling scale
    batches_per_epoch = len(trainloader)
    num_epochs = 800 #epochs * 4 # 200 * 4 =800
    print(f"Artificial epochs end point: {num_epochs}")
    print(f"Init_lr = {init_lr}")
    save_path = HydraConfig.get().runtime.output_dir
    log_dir = os.path.join(save_path, "logger")
    print(f"Logger workdir: {log_dir}")
    writer = SummaryWriter(log_dir= log_dir)
    # parameters grouping change if constant_predictor_lr is wanted on predictor 
    optimizer = get_optimizer(model=net, lr=init_lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay, name='sgd')
    # optimizer = torch.optim.SGD(net.parameters(), lr=init_lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    lr_scheduler = LR_Scheduler(optimizer=optimizer, warmup_epochs=cfg.optimizer.warmup_epochs, warmup_lr=cfg.optimizer.warmup_lr * bs / 256, 
                                num_epochs=num_epochs, base_lr=init_lr, final_lr=cfg.optimizer.final_lr * bs / 256, iter_per_epoch=batches_per_epoch,
                                constant_predictor_lr=False) # see the end of SimSiam paper section 4.2 predictor     
    # return 0
    # train loop SSL
    info_dict = train_loop_ssl(net=net, trainloader=trainloader, testloader=testloader, memoryloader=memoryloader, optimizer=optimizer, epochs=epochs, 
                               device=DEVICE, init_lr=init_lr, lr_scheduler=lr_scheduler, writer=writer)
    
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    max_mem = torch.cuda.max_memory_allocated()
    print(f"Max memory allocated: {max_mem / 1024**2} MB")
    writer.close()
    # Plot and save results
    plot_results_SSL(save_path=save_path, info_dict=info_dict, subset=subset, num_classes=n_classes, model=model)
    # Save the current net module 
    if not os.path.exists(cfg.checkpoint_path):
            os.makedirs(cfg.checkpoint_path)           
    prefix_net: str = f"{cfg.checkpoint_path}centr_pretrained_{init_lr}_{model}_classes{n_classes}_E{epochs}.pth"
    torch.save(net.state_dict(), prefix_net)
    print("Results saved, exiting succesfully...")
    print(f"---------Experiment Completed in : {(time.time()-start)/60} minutes")
if __name__ == "__main__":
    # parser
    # import cProfile
    # cProfile.run('main()', 'profiles/centr_simsiam_fileE=1_Nwork=4_k_2')
    main()
    # snakeviz.
    pass