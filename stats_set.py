import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms
import torch
from dataset_prep import *
from dataset import *
import pandas as pd


def food101_mean_std(trainset):
    imgs = [item[0] for item in trainset] # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()

    return mean_r, mean_g, mean_b, std_r, std_g, std_b

def main():
    # datapath = 'D:/DesktopC/Datasets/data/'
    datapath = './data'
    trf = transforms.Compose([
            Resize_with_pad(),
            transforms.ToTensor(),
        ])
    # subset = [False, True]
    # n_classes = [4,10,30]
    # trainset = get_food101()
    print("Loading data...")
    trainset, testset = get_food101(transform=trf, datapath=datapath, subset=False)
    # trainset = datasets.Food101(root=datapath, split="train", transform=trf, download= True)
    # testset = datasets.Food101(root=datapath, split="test", transform=trf, download= True)
    #becareful of distortion
    print("Calculating dataset statistics....")
    mean_r, mean_g, mean_b, std_r, std_g, std_b = food101_mean_std(trainset)
    print(f'Scaled Mean Pixel Value (R G B):{mean_r,mean_g,mean_b},\nScaled Pixel Value Std (R G B):{std_r, std_g, std_b}')
    meanRGB = np.array([mean_r,mean_g,mean_b])
    stdRGB = np.array([std_r, std_g, std_b])
    df = pd.DataFrame(
        {"meanRGB": meanRGB, "stdRGB": stdRGB}
    )
    file_name = "food101_stats.csv"
    # os.path.join(
    #     save_path,
    #     f"{file_suffix}.csv",
    # )
    df.to_csv(file_name,index=False)

if __name__ == "__main__":
    main()
    