import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
# from model import *

#MODEL UTILS
def save_model(net, model, subset):
    if subset:
         datamode = "subset"
    else:
         datamode = "full"
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')
    PATH = f"./saved_models/food101{datamode}_{model}.pth"
    torch.save(net.state_dict(), PATH)
    print(f"Model saved to: {PATH}")

def copy_model(net, model, datamode):
    try:
            print("warm start")
            PATH = f"./saved_models/food101{datamode}_{model}.pth"
            net.load_state_dict(torch.load(PATH))
            print(f"Warm start - {model} model loaded")
    except:
            print("No model available for load. Continuing with fresh start")

#IMAGE PRINTERS
def denormalize(images, imgenet_std, imgenet_mean):
    means = torch.tensor(imgenet_mean).reshape(1, 3, 1, 1)
    stds = torch.tensor(imgenet_std).reshape(1, 3, 1, 1)
    return images * stds + means

def show_batch(dl, imgenet_std, imgenet_mean, bs):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denorm_images = denormalize(images, imgenet_std, imgenet_mean)
        ax.imshow(torchvision.utils.make_grid(denorm_images[:int (bs/4)], nrow=8).permute(1, 2, 0).clamp(0,1))
        if not os.path.exists('./images'):
            os.makedirs('./images') 
        plt.savefig("./images/batch_part_grid", bbox_inches='tight')
        plt.close()
        break

#TRAINING RESULTS MANAGER
def plot_results(save_path: str, info_dict: Dict[str, List], subset: bool, num_classes: int, model: str):
    if subset:
         datamode = "subset"
    else:
         datamode = "full"
    results = pd.DataFrame(info_dict)
    # file_suffix = f"stats_food101{datamode}_{model}"
    file_name = os.path.join(
        save_path,
        f"centr_food_{datamode}_{num_classes}classes_{model}.csv",
    )
    results.to_csv(file_name, index=False)
    keys = info_dict.keys()
    values = info_dict.values()
    accs = []
    losses =[]
    for i, (key, value) in enumerate(zip(keys,values)):
        if i == 0: 
            losses = np.array(value)
        if i == 2:
            losses = np.stack((losses, np.array(value)), axis=0)
        
        if i == 1:
            accs = np.array(value)
        if i == 3:
            accs = np.stack((accs, np.array(value)), axis=0)

    if not os.path.exists('./images'):
        os.makedirs('./images') 

    xset = [i+1 for i in range(len(accs[0]))]
    figname1 = f"./images/accuracies_{datamode}_{num_classes}classes_{model}.png"
    plt.plot(xset, accs[0], 'x-', xset, accs[1], 'x-')
    plt.legend(('Train Acc', 'Test Acc'), loc = 'lower center')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.savefig(figname1, bbox_inches='tight')
    plt.close()

    figname2 = f"./images/losses_{datamode}_{num_classes}classes_{model}.png"
    plt.plot(xset, losses[0], 'x-', xset, losses[1], 'x-')
    plt.legend(('Train Loss', 'Test Loss'), loc = 'lower center')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(figname2, bbox_inches='tight')
    plt.close()