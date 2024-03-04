from random import Random
import torch
from torch.utils.data import random_split, Dataset, Subset
from typing import List
import numpy as np
import torchvision.transforms.functional as F

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

class Resize_with_pad:
    """Class transformation to upscale a image to HxW using zero-padding"""
    def __init__(self, w=512, h=512):
        self.w = w
        self.h = h

    def __call__(self, image):

        w_1, h_1 = image.size
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1


        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 0, "constant")
                return F.resize(image, [self.h, self.w])

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])

        else:
            return F.resize(image, [self.h, self.w])

def partitioning_iid(trainset, num_partitions: int, balance: bool = True, seed: int = 2024):
    """Partitionining according to IID and random permutation.

    Parameters
    ----------
    trainset: Dataset
        Dataset to split into partitions 
    num_partitions : int
        The number of clients that hold a part of the data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 2024

    Returns
    -------
    List[CustomSubset]
    The list of datasets for each client
    """
    ## IID Sampling 
    # [00..00 | 11..11 | 22..22 | 33..33] = > 
    if balance:
        # na valw hard coded ta sets gia 
        # to ypoloipo ths // na to dinw se kapoion client den prz
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        # or 
        # last client collects the modulo 
        # idxs dict -> Client ID : Client's indexes
        idxs_map = {}
        # range sequence indices
        idxs = np.array(range(len(trainset.indices)))
        # get the targets
        tmp_t = trainset.labels
        # unique labels
        num_classes = len(set(tmp_t))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[tmp_t == i])
        # every client has the same classes
        class_num_per_client = [num_classes for _ in range(num_partitions)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_partitions):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_partitions / num_classes) * num_classes))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            # fair balanced split
            num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            num_samples.append(num_all_samples-sum(num_samples))
            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in idxs_map.keys():
                    idxs_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    idxs_map[client] = np.append(idxs_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

        # range sequence indices into dataset indices reference
        idxs_map = [trainset.indices[idxs_map[client]] for client in range(len(idxs_map))]
        # assign custom trainsets
        trainsets = [CustomSubset(trainset.dataset, idxs_map[i]) for i in range(len(idxs_map))]

    else: 
        # split trainset into `num_partitions` trainsets (one per client)
        # figure out number of training examples per partition
        num_images = len(trainset) // num_partitions
        # a list of partition lengths (all partitions are of equal size)
        partition_len = [num_images] * num_partitions
        idxs = list(range(len(trainset.indices)))
        np.random.seed(seed)
        np.random.shuffle(idxs)
        trainsets_idx = [idxs[id*p_len:(id+1)*p_len] for id,p_len in enumerate(partition_len)]
        idxs_map = trainset.indices[trainsets_idx]
        trainsets = [CustomSubset(trainset.dataset, idxs_map[i]) for i in range(len(idxs_map))]
    
    return trainsets

def partitioning_dirichlet(alpha, trainset, num_partitions: int, seed: int = 2024):
    """Partitionining according to the Dirichlet distribution.

    Parameters
    ----------
    alpha: float
        Parameter of the Dirichlet distribution
    trainset: Dataset
        Dataset to split into partitions 
    num_partitions : int
        The number of clients that hold a part of the data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 2024

    Returns
    -------
    List[CustomSubset]
    The list of datasets for each client
    """
    # DIRICHLET SAMPLING
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(seed)

    # get the targets
    tmp_t = trainset.labels
    # na dw gia plhres dataset
    # tmp_t = trainset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    while min_samples < min_required_samples_per_client:
        idx_clients: List[List] = [[] for _ in range(num_partitions)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_partitions))
            # balancing
            proportions = np.array(
                [
                    p * (len(idx_j) < total_samples / num_partitions)
                    for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

    idxs_map = [trainset.indices[idx_clients[i]] for i in range(num_partitions)]
    trainsets = [CustomSubset(trainset.dataset, idxs_map[i]) for i in range(len(idxs_map))]
    # trainsets = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets

# plot thn katanomh se clients 
#TBD: similarity splitting, power law population splitting, min label quantity per client split

