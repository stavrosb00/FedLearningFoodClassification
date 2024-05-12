import torch
from torch.utils.data import random_split, DataLoader, Subset, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose, RandomResizedCrop, RandomApply, ColorJitter, RandomGrayscale, RandomHorizontalFlip, Resize
import torchvision.transforms.functional as F
from torchvision.datasets import Food101
from dataset_prep import *
from typing import List, Tuple
from omegaconf import DictConfig
import torchvision
from utils import plot_exp_summary, plot_client_stats, get_subset_stats
import os
from sklearn.model_selection import train_test_split

class CustomSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        # labels_hold = np.ones(len(dataset)) *999 #( some number not present in the #labels just to make sure)
        # labels_hold[self.indices] = labels 
        # np_tr_idx = np.array(self.indices)
        np_tr_lab = np.array(dataset._labels)
        tr_mapped_lab = np_tr_lab[indices]
        self.labels = tr_mapped_lab
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.labels[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)
    
# from torchvision import transforms
def get_food101(transform, datapath: str = 'D:/DesktopC/Datasets/data/', subset: bool = True, num_classes: int = 4):
    """Download Food101 and apply transformation."""
    trainset = Food101(root=datapath, split="train", transform=transform, download= True)
    testset = Food101(root=datapath, split="test", transform=transform, download= True)
    # trainset_labels = trainset.classes
    if subset:
        #Taking Subset of trainset and testset
        # select classes you want to include in your subset
        list = [i for i in range(num_classes)]
        classes = torch.tensor(list)
        # classes = torch.tensor([0, 1, 2, 3])
        # get indices that correspond to one of the selected classes
        train_indices = (torch.tensor(trainset._labels)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
        np_tr_idx = np.array(train_indices)
        # np_tr_lab = np.array(trainset._labels)
        # tr_mapped_lab = np_tr_lab[np_tr_idx]
        # subset the dataset
        train_sub = CustomSubset(trainset, np_tr_idx) # tr_mapped_lab)
        # train_sub = Subset(trainset, train_indices)

        # get indices that correspond to one of the selected classes
        test_indices = (torch.tensor(testset._labels)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
        np_test_idx = np.array(test_indices)
        # np_test_lab = np.array(testset._labels)
        # test_mapped_lab = np_test_lab[np_test_idx]
        # subset the dataset
        test_sub = CustomSubset(testset, np_test_idx) #, test_mapped_lab)
        # test_sub = Subset(testset, test_indices)
        return train_sub, test_sub
    else:
        raise NotImplementedError('You need to pick subset=True, even for num_classes=10,50,101 etc.')
        # return trainset, testset                
          
def load_dataset(datapath: str, 
                 subset: bool,
                 num_classes: int,
                 num_workers: int,
                 num_partitions: int, 
                    batch_size: int,
                    partitioning: str = "iid", 
                    alpha: float = 0.5,
                    balance: bool = True,
                    seed: int = 2024,
                    val_ratio: float = 0.3) -> tuple[list[DataLoader], list[DataLoader], DataLoader]:
    """Download Food101 and generate partitions & loaders for federating learning."""
    print("Loading data...")
    #transformation based on imagenet & resnet18 settings  or from dataset normalization stats
    # trf = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    trf = Compose([
         Resize_with_pad(),
         ToTensor(),
    ])
    trainset, testset = get_food101(trf, datapath,subset, num_classes)
    if partitioning == "iid":
        trainsets = partitioning_iid(trainset, num_partitions, balance, seed)
        title_str = f"Clients data partitioning: {partitioning.upper()}"
        if balance:
            save_str_cid: str ="balanced" # equal splits per client based on each label quantity
        else:
            save_str_cid: str ="U" #uniform shuffle rand generator
        save_str_exp = f"images/clients_vis/{partitioning}/clients_{len(trainsets)}/classes_{num_classes}/{save_str_cid}/summary" #clients_{len(trainsets)}_classes_{num_classes}_
    elif partitioning == "dirichlet":
        trainsets = partitioning_dirichlet(alpha, trainset, num_partitions, seed)
        title_str = f"Clients data partitioning: {partitioning.upper()}, a={alpha}"
        save_str_cid: str = (f"a_{alpha}")
        save_str_exp = f"images/clients_vis/{partitioning}/clients_{len(trainsets)}/classes_{num_classes}/{save_str_cid}/summary" #alpha_{alpha}_clients_{len(trainsets)}_classes_{num_classes}_
    else:
        raise NotImplementedError(f"{partitioning} partitioning not done")


    # Obtain and save data statistic plots
    if not os.path.exists(f'./images/clients_vis/{partitioning}/clients_{len(trainsets)}/classes_{num_classes}/{save_str_cid}'):
            os.makedirs(f'./images/clients_vis/{partitioning}/clients_{len(trainsets)}/classes_{num_classes}/{save_str_cid}')
    plot_exp_summary(trainsets, title_str, num_classes, save_str_exp)
    for c_id, sub_trainset in enumerate(trainsets):
        tmp = get_subset_stats(sub_trainset)
        plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid, save_str_exp)
    
    # trainsets on IID case if balance=False
    # create dataloaders with train+val support
    trainloaders: list[CustomSubset] = []
    valloaders: list[CustomSubset] = []
    np.random.seed(seed)
    for c_id, trainset_ in enumerate(trainsets):
        if balance:
            if partitioning =="iid":
                train_indices, val_indices = train_test_split(trainset_.indices, test_size=val_ratio, stratify=trainset_.labels)
                
                trainloaders.append(CustomSubset(trainset_.dataset, train_indices))
                valloaders.append(CustomSubset(trainset_.dataset, val_indices))
                tmp = get_subset_stats(trainloaders[-1])
                plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_train", save_str_exp+ "_train", split="train")
                tmp = get_subset_stats(valloaders[-1])
                plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_val", save_str_exp+ "_val", split="val")
            else: #dirichlet
                # Get unique class labels and their counts
                unique_labels, class_counts = np.unique(trainset_.labels, return_counts=True)
                # Define a minimum threshold for the number of samples in each class
                min_samples_per_class = 2
                # Filter out classes with too few samples
                filtered_classes = [label for label, count in zip(unique_labels, class_counts) if count >= min_samples_per_class]
                # Filter indices corresponding to the filtered classes
                filtered_indices = [index for index, label in enumerate(trainset_.labels) if label in filtered_classes]
                # Split filtered (X,y)
                train_indices, val_indices = train_test_split(trainset_.indices[filtered_indices], test_size=val_ratio, stratify=trainset_.labels[filtered_indices])
                # Get the excluded indices
                excluded_indices = [index for index in range(len(trainset_.labels)) if index not in filtered_indices]
                train_indices = np.concatenate((train_indices, np.array(trainset_.indices[excluded_indices])))

                trainloaders.append(CustomSubset(trainset_.dataset, train_indices))
                valloaders.append(CustomSubset(trainset_.dataset, val_indices))
                tmp = get_subset_stats(trainloaders[-1])
                plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_train", save_str_exp+ "_train", split="train")
                tmp = get_subset_stats(valloaders[-1])
                plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_val", save_str_exp+ "_val", split="val")
        else: #uni split if not balanced split requested
            num_total = len(trainset_)
            num_val = int(val_ratio * num_total)
            num_train = num_total - num_val
            
            # choose validation indexes
            choices = np.random.choice(range(num_total), size=num_val, replace=False)
            # boolean split
            idxs_val = np.zeros(num_total, dtype=bool)
            idxs_val[choices] = True
            idxs_tr = ~idxs_val
            # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
            trainloaders.append(CustomSubset(trainset_.dataset, trainset_.indices[idxs_tr]))
            valloaders.append(CustomSubset(trainset_.dataset, trainset_.indices[idxs_val]))
            tmp = get_subset_stats(trainloaders[-1])
            plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_train", save_str_exp+ "_train", split="train")
            tmp = get_subset_stats(valloaders[-1])
            plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_val", save_str_exp+ "_val", split="val")

    ##### possible plot on splitted sets after for loop #####
    #
    # reproducable shuffling for training
    # torch.manual_seed(seed)
    # G = torch.Generator()
    # G.manual_seed(seed)
    # generator=G,
    # construct data loaders to their respective list
    trainloaders = [DataLoader(Subset(trainloaders[i].dataset, trainloaders[i].indices), batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers) 
                         for i in range(len(trainloaders))]
    valloaders = [DataLoader(Subset(valloaders[i].dataset, valloaders[i].indices), batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers) 
                         for i in range(len(valloaders))]
    # testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    testloader = DataLoader(Subset(testset.dataset, testset.indices), batch_size=batch_size, num_workers=num_workers)
    return trainloaders, valloaders, testloader

def load_dataset_SSL(datapath: str, 
                 subset: bool,
                 num_classes: int,
                 num_workers: int,
                 num_partitions: int, 
                    batch_size: int,
                    partitioning: str = "iid", 
                    alpha: float = 0.5,
                    balance: bool = True,
                    seed: int = 2024,
                    val_ratio: float = 0.0,
                    rad_ratio: float = 0.02) -> tuple[list[DataLoader], list[DataLoader], DataLoader, DataLoader, DataLoader]:
    """Download Food101 and generate partitions & loaders for federating self-supervised learning."""
    augmentation = Compose([
        RandomResizedCrop(256, scale=(0.2, 1.), interpolation= F.InterpolationMode.BICUBIC),
        RandomApply([
            ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        RandomGrayscale(p=0.2),
        RandomHorizontalFlip(),
        ToTensor(),
        # normalize
    ])
    simple_trf = Compose([
        Resize((256, 256), interpolation= F.InterpolationMode.BICUBIC),
        ToTensor(),
    ])
    trainset, testset, memoryset = get_food101_ssl(augmentation, simple_trf, datapath, subset, num_classes)

    if partitioning == "iid":
        trainsets = partitioning_iid(trainset, num_partitions, balance, seed)
        title_str = f"Clients data partitioning: {partitioning.upper()}"
        if balance:
            save_str_cid: str ="balanced" # equal splits per client based on each label quantity
        else:
            save_str_cid: str ="U" #uniform shuffle rand generator
        save_str_exp = f"images/clients_vis/{partitioning}/clients_{len(trainsets)}/classes_{num_classes}/{save_str_cid}/summary" #clients_{len(trainsets)}_classes_{num_classes}_
    elif partitioning == "dirichlet":
        trainsets = partitioning_dirichlet(alpha, trainset, num_partitions, seed)
        title_str = f"Clients data partitioning: {partitioning.upper()}, a={alpha}"
        save_str_cid: str = (f"a_{alpha}")
        save_str_exp = f"images/clients_vis/{partitioning}/clients_{len(trainsets)}/classes_{num_classes}/{save_str_cid}/summary" #alpha_{alpha}_clients_{len(trainsets)}_classes_{num_classes}_
    else:
        raise NotImplementedError(f"{partitioning} partitioning not done")


    # Obtain and save data statistic plots
    if not os.path.exists(f'./images/clients_vis/{partitioning}/clients_{len(trainsets)}/classes_{num_classes}/{save_str_cid}'):
            os.makedirs(f'./images/clients_vis/{partitioning}/clients_{len(trainsets)}/classes_{num_classes}/{save_str_cid}')
    plot_exp_summary(trainsets, title_str, num_classes, save_str_exp)
    for c_id, sub_trainset in enumerate(trainsets):
        tmp = get_subset_stats(sub_trainset)
        plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid, save_str_exp)
    
    # trainsets on IID case if balance=False
    # create dataloaders with train+val support
    trainloaders: list[CustomSubset] = []
    valloaders: list[CustomSubset] = []
    np.random.seed(seed)
    for c_id, trainset_ in enumerate(trainsets):
        if balance and val_ratio != 0:
            if partitioning =="iid":
                train_indices, val_indices = train_test_split(trainset_.indices, test_size=val_ratio, stratify=trainset_.labels)
                
                trainloaders.append(CustomSubset(trainset_.dataset, train_indices))
                valloaders.append(CustomSubset(trainset_.dataset, val_indices))
                tmp = get_subset_stats(trainloaders[-1])
                plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_train", save_str_exp+ "_train", split="train")
                tmp = get_subset_stats(valloaders[-1])
                plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_val", save_str_exp+ "_val", split="val")
            else: #dirichlet
                # Get unique class labels and their counts
                unique_labels, class_counts = np.unique(trainset_.labels, return_counts=True)
                # Define a minimum threshold for the number of samples in each class
                min_samples_per_class = 2
                # Filter out classes with too few samples
                filtered_classes = [label for label, count in zip(unique_labels, class_counts) if count >= min_samples_per_class]
                # Filter indices corresponding to the filtered classes
                filtered_indices = [index for index, label in enumerate(trainset_.labels) if label in filtered_classes]
                # Split filtered (X,y)
                train_indices, val_indices = train_test_split(trainset_.indices[filtered_indices], test_size=val_ratio, stratify=trainset_.labels[filtered_indices])
                # Get the excluded indices
                excluded_indices = [index for index in range(len(trainset_.labels)) if index not in filtered_indices]
                train_indices = np.concatenate((train_indices, np.array(trainset_.indices[excluded_indices])))

                trainloaders.append(CustomSubset(trainset_.dataset, train_indices))
                valloaders.append(CustomSubset(trainset_.dataset, val_indices))
                tmp = get_subset_stats(trainloaders[-1])
                plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_train", save_str_exp+ "_train", split="train")
                tmp = get_subset_stats(valloaders[-1])
                plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_val", save_str_exp+ "_val", split="val")
        else: #uni split if not balanced split requested
            num_total = len(trainset_)
            num_val = int(val_ratio * num_total)
            num_train = num_total - num_val
            
            # choose validation indexes
            choices = np.random.choice(range(num_total), size=num_val, replace=False)
            # boolean split
            idxs_val = np.zeros(num_total, dtype=bool)
            idxs_val[choices] = True
            idxs_tr = ~idxs_val
            # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
            trainloaders.append(CustomSubset(trainset_.dataset, trainset_.indices[idxs_tr]))
            tmp = get_subset_stats(trainloaders[-1])
            plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_train", save_str_exp+ "_train", split="train")
            if val_ratio !=0:
                valloaders.append(CustomSubset(trainset_.dataset, trainset_.indices[idxs_val]))
                tmp = get_subset_stats(valloaders[-1])
                plot_client_stats(partitioning, c_id+1, tmp, num_classes, save_str_cid + "_val", save_str_exp+ "_val", split="val")

    # construct data loaders to their respective list
    trainloaders = [DataLoader(Subset(trainloaders[i].dataset, trainloaders[i].indices), batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers) 
                         for i in range(len(trainloaders))]
    if val_ratio !=0:
        valloaders = [DataLoader(Subset(valloaders[i].dataset, valloaders[i].indices), batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers) 
                            for i in range(len(valloaders))]
    else:
        valloaders = [None] * num_partitions
    # testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    testloader = DataLoader(Subset(testset.dataset, testset.indices), batch_size=batch_size, num_workers=num_workers)
    # Load representation alignment dataset which is publicly loaded from clients
    train_indices, val_indices = train_test_split(trainset.indices, test_size=rad_ratio, stratify=trainset.labels)
    # radset = CustomSubset(trainset.dataset, val_indices)
    radset = CustomSubset(memoryset.dataset, val_indices)
    # If I want diversify the augmentation of RAD.
    # radset = Food101(root=datapath, split="train", transform=augmentation, download= True)
    # radset = CustomSubset(radset, val_indices)

    # radloader = DataLoader(radset, batch_size=batch_size, num_workers=num_workers)
    
    # print(trainset.dataset.transform)
    # radset.dataset.transform = augmentation
    # print(radset.dataset.transform)
    # print(trainset.dataset.transform)
    rad_stats = get_subset_stats(radset)
    print(rad_stats)
    radloader = DataLoader(Subset(radset.dataset, radset.indices), batch_size=batch_size, num_workers=num_workers)
    # memoryloader-artificial knowledge for monitoring 
    memoryloader = DataLoader(Subset(memoryset.dataset, memoryset.indices), batch_size=batch_size, num_workers=num_workers) 
    return trainloaders, valloaders, testloader, memoryloader, radloader

def load_centr_data(datapath: str, 
                 subset: bool,
                 num_classes: int,
                 num_workers: int,
                    batch_size: int):
    """Download Food101 dataset for centralised learning."""
    print("Loading data...")
    #transformation based on imagenet & resnet18 settings  or from dataset normalization stats
    # trf = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    trf = Compose([
         Resize_with_pad(),
         ToTensor(),
    ])
    trainset, testset = get_food101(trf, datapath,subset, num_classes)
    # Subset(testset.dataset, testset.indices)
    return DataLoader(Subset(trainset.dataset, trainset.indices), batch_size=batch_size, shuffle=True, num_workers=num_workers), DataLoader(Subset(testset.dataset, testset.indices), batch_size=batch_size, num_workers=num_workers)

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def get_food101_ssl(transform, simple_trf, datapath: str = 'D:/DesktopC/Datasets/data/', subset: bool = True, num_classes: int = 4):
    """Download Food101 for SSL and apply augmentation/transformation."""
    trainset = Food101(root=datapath, split="train", transform=TwoCropsTransform(transform), download= True)
    testset = Food101(root=datapath, split="test", transform=simple_trf, download= True)
    memoryset = Food101(root=datapath, split="train", transform=simple_trf, download= True)
    # print(type(trainset))
    # print(type(testset))
    if subset:
        #Taking Subset of trainset and testset
        # select classes you want to include in your subset
        list = [i for i in range(num_classes)]
        classes = torch.tensor(list)
        # get indices that correspond to one of the selected classes
        train_indices = (torch.tensor(trainset._labels)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
        np_tr_idx = np.array(train_indices)
        # subset the dataset
        train_sub = CustomSubset(trainset, np_tr_idx) # tr_mapped_lab)
        # get indices that correspond to one of the selected classes
        test_indices = (torch.tensor(testset._labels)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
        np_test_idx = np.array(test_indices)
        # subset the dataset
        test_sub = CustomSubset(testset, np_test_idx) #, test_mapped_lab)
        # test_sub = Subset(testset, test_indices)
        memory_indices = (torch.tensor(memoryset._labels)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
        np_memory_idx = np.array(memory_indices)
        memory_sub = CustomSubset(memoryset, np_memory_idx)
        return train_sub, test_sub, memory_sub
    else:
        raise NotImplementedError('You need to pick subset=True, even for num_classes=10,50,101 etc.')

def load_centr_data_SSL(datapath: str, 
                 subset: bool,
                 num_classes: int,
                 num_workers: int,
                    batch_size: int):
    """Download Food101 dataset for centralised self-supervised learning."""
    print("Loading data...")
    #transformation based on imagenet & resnet18 settings  or from dataset normalization stats
    # trf = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    # normalize = Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    augmentation = Compose([
        RandomResizedCrop(256, scale=(0.2, 1.), interpolation= F.InterpolationMode.BICUBIC),
        RandomApply([
            ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        RandomGrayscale(p=0.2),
        RandomHorizontalFlip(),
        ToTensor(),
        # normalize
    ])
    simple_trf = Compose([
        Resize((256, 256), interpolation= F.InterpolationMode.BICUBIC),
        ToTensor(),
    ])
    trainset, testset, memoryset = get_food101_ssl(augmentation, simple_trf, datapath, subset, num_classes)
    train_loader = DataLoader(Subset(trainset.dataset, trainset.indices), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(Subset(testset.dataset, testset.indices), batch_size=batch_size, num_workers=num_workers)
    memory_loader = DataLoader(Subset(memoryset.dataset, memoryset.indices), batch_size=batch_size, num_workers=num_workers)
    # Subset(testset.dataset, testset.indices)
    return train_loader, test_loader, memory_loader












# def get_mnist(data_paths: str = './data'):
#     """Download MNIST and apply minimal transformation."""

#     tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
#     trainset = MNIST(data_paths, train=True, download =True, transform=tr)
#     testset = MNIST(data_paths, train=False, download =True, transform=tr)

#     return trainset, testset

# def prepare_dataset(num_partitions: int, 
#                     batch_size: int, 
#                     val_ratio: float = 0.1):
#     """Download MNIST and generate IID partitions."""

#     # download MNIST in case it's not already in the system
#     trainset, testset = get_mnist()

    
#     # split trainset into `num_partitions` trainsets (one per client)
#     # figure out number of training examples per partition
#     num_images = len(trainset) // num_partitions

#     # a list of partition lenghts (all partitions are of equal size)
#     partition_len = [num_images] * num_partitions
#     trainsets = random_split(
#         trainset, partition_len, torch.Generator().manual_seed(2023)
#     )

#     # create dataloaders with train+val support
#     trainloaders = []
#     valloaders = []
#     for trainset_ in trainsets:
#         num_total = len(trainset_)
#         num_val = int(val_ratio * num_total)
#         num_train = num_total - num_val

#         for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))
#         # construct data loaders and append to their respective list.
#         # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
#         trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
#         valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))
        
#     # We leave the test set intact (i.e. we don't partition it)
#     # This test set will be left on the server side and we'll be used to evaluate the
#     # performance of the global model after each round.
#     # Please note that a more realistic setting would instead use a validation set on the server for
#     # this purpose and only use the testset after the final round.
#     # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
#     # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
#     # in main.py above the strategy definition for more details on this)
#     testloader = DataLoader(testset, batch_size=128)

#     return trainloaders, valloaders, testloader

# def load_datasets(config: DictConfig) -> Tuple[List[DataLoader], DataLoader, List]:
#     """Create the dataloaders to be fed into the model.

#     Parameters
#     ----------
#     config: DictConfig
#         Parameterises the dataset partitioning process
#     num_clients : int
#         The number of clients that hold a part of the data
#     val_ratio : float, optional
#         The ratio of training data that will be used for validation (between 0 and 1),
#         by default 0.1
#     batch_size : int, optional
#         The size of the batches to be fed into the model, by default 32
#     seed : int, optional
#         Used to set a fix seed to replicate experiments, by default 42

#     Returns
#     -------
#     Tuple[DataLoader, DataLoader, List]
#         The DataLoader for training, the DataLoader for testing, client dataset sizes.
#     """
#     # download MNIST in case it's not already in the system
#     trainset, testset = get_mnist()

#     partition_sizes = [1.0 / config.num_clients for _ in range(config.num_clients)]

#     partition_obj = DataPartitioner(
#         trainset, partition_sizes, is_non_iid=config.NIID, alpha=config.alpha
#     )
#     ratio = partition_obj.ratio

#     trainloaders = []
#     for data_split in range(config.num_clients):
#         client_partition = partition_obj.use(data_split)
#         trainloaders.append(
#             torch.utils.data.DataLoader(
#                 client_partition,
#                 batch_size=config.batch_size,
#                 shuffle=True,
#             )
#         )

#     test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

#     return trainloaders, test_loader, ratio

# @hydra.main(config_path="conf", config_name="base", version_base=None)
# def main(cfg: DictConfig):
    # print(cfg.datapath)


# if __name__ == "__main__":

    
    # datapath = 'D:/DesktopC/Datasets/data/'
    # subset = True
    # num_classes = 4
    # num_workers = 1
    # batch_size=64
    # load_centr_data(datapath=datapath, subset=subset, num_classes=num_classes, num_workers=num_workers, )



# trf2 = Compose([
    #     torchvision.transforms.Resize((298,224)),
    #     ToTensor(),
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])