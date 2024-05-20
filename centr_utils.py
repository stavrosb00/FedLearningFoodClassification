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
        # denorm_images = denormalize(images, imgenet_std, imgenet_mean)
        denorm_images = images
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

def plot_results_downstream(save_path: str, info_dict: Dict[str, List], subset: bool, num_classes: int, model: str):
    if subset:
         datamode = "subset"
    else:
         datamode = "full"
    results = pd.DataFrame(info_dict) #, index=range(len(info_dict)))
    # file_suffix = f"stats_food101{datamode}_{model}"
    file_name = os.path.join(
        save_path,
        f"centr_food_{datamode}_{num_classes}classes_{model}.csv",
    )
    results.to_csv(file_name, index=False)
    accs = []
    accs.append(info_dict["train_acc"])
    accs.append(info_dict["test_acc"])
    losses = []
    losses.append(info_dict["train_loss"])
    losses.append(info_dict["test_loss"])
    xset =[x+1 for x in range(len(accs[0]))]
    figname1 = f"./images/accuracies_{datamode}_{num_classes}classes_{model}.png"
    plt.plot(xset, accs[0], '-', xset, accs[1], '-')
    plt.legend(('Train Acc', 'Test Acc'), loc = 'lower center')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.savefig(figname1, bbox_inches='tight')
    plt.close()

    figname2 = f"./images/losses_{datamode}_{num_classes}classes_{model}.png"
    plt.plot(xset, losses[0], '-')
    #, xset, losses[1], '-')
    plt.legend('Train Loss') 
    #, 'Test Loss'), loc = 'lower center')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(figname2, bbox_inches='tight')
    plt.close()

def plot_results_SSL(save_path: str, info_dict: Dict[str, List], subset: bool, num_classes: int, model: str):
    if subset:
         datamode = "subset"
    else:
         datamode = "full"
    results = pd.DataFrame(info_dict) #, index=range(len(info_dict)))
    # file_suffix = f"stats_food101{datamode}_{model}"
    file_name = os.path.join(
        save_path,
        f"centr_SSL_food_{datamode}_{num_classes}classes_{model}.csv",
    )
    results.to_csv(file_name, index=False)

    accs = info_dict["knn_accuracy"]
    xset =[x+1 for x in range(len(accs))]
    figname = f"./images/knn_accuracy_{datamode}_{num_classes}classes_{model}.png"
    plt.plot(xset, accs, '-')
    plt.xlabel('Epoch')
    plt.ylabel('kNN Accuracy %')
    plt.title('Centralised pretraining SimSiam')
    # plt.grid(True)
    # plt.show()
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

    losses = info_dict["train_loss"]
    figname2 = f"./images/d_losses_{datamode}_{num_classes}classes_{model}.png"
    plt.plot(xset, losses, '-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Centralised pretraining SimSiam')
    plt.savefig(figname2, bbox_inches='tight')
    plt.close()

def plot_results_SSL_csv(csv_file: str):
    """Plot train loss and kNN accuracy

    Args:
        csv_file (str): CSV file string from centralised experiment's saved results

    Raises:
        ValueError: CSV file origin must have the two corresponding columns to kNN accuracy and train loss
    """
    df = pd.read_csv(csv_file)
    
    # Check if the required columns are present in the dataframe
    if 'knn_accuracy' not in df.columns or 'train_loss' not in df.columns:
        raise ValueError("The CSV file must contain 'knn_accuracy' and 'train_loss' columns.")
    
    # Extract the 'knn_accuracy' and 'train_loss' columns
    knn_accuracy = df['knn_accuracy']
    train_loss = df['train_loss']
    
    # Plot the extracted columns using matplotlib
    plt.figure(figsize=(10, 5))
    figname = f"./images/knn_accuracy_train_loss_centralised_simsiam.png"
    # Plot knn_accuracy
    plt.subplot(1, 2, 1)
    plt.plot(knn_accuracy, label='kNN Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.title('kNN Accuracy over Epochs')
    plt.legend()

    # Plot train_loss
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train D Loss over epochs')
    plt.legend()
    
    # Show the plots
    plt.savefig(figname, bbox_inches='tight')
    # plt.tight_layout()
    # plt.show()