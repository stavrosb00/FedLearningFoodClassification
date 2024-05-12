import torch
import numpy as np
import math
from sklearn.metrics.pairwise import rbf_kernel
import time
from dataset import *
# Hetero-SSFL functions related to Linear-CKA distance metric

# https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python
# https://discuss.pytorch.org/t/torch-is-slow-compared-to-numpy/117502 ndarrays format serialized anyways on Flower communication protocol
# https://datumorphism.leima.is/cards/machine-learning/measurement/centered-kernel-alignment/ 
def linear_CKA_fast(X, Y):
    # Compute the Gram matrices
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    
    # Center the Gram matrices
    n = X.shape[0]
    unit = np.ones((n, n))
    I = np.eye(n)
    H = I - unit / n
    
    L_X_centered = np.dot(np.dot(H, L_X), H)
    L_Y_centered = np.dot(np.dot(H, L_Y), H)
    
    # Compute the linear CKA
    hsic = np.sum(L_X_centered * L_Y_centered)
    var1 = np.sqrt(np.sum(L_X_centered * L_X_centered))
    var2 = np.sqrt(np.sum(L_Y_centered * L_Y_centered))

    return hsic / (var1 * var2)

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the same with one time centering
    # return np.dot(H, K)  # KH

# https://en.wikipedia.org/wiki/Radial_basis_function_kernel
def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


if __name__ == "__main__":
    print("Testing things")
    K = torch.randn(150, 2048)
    K2 = torch.randn(150, 2048)
    K_np = K.cpu().numpy()
    K2_np = K2.cpu().numpy()
    start =time.time()
    print('Linear CKA, between X and Y: {}'.format(linear_CKA(K_np, K2_np)))
    print(time.time() - start)

    start =time.time()
    print('Linear CKA, between X and Y: {}'.format(kernel_CKA(K_np, K2_np)))
    print(time.time() - start)

    start3 =time.time()
    print('Linear CKA, between X and Y: {}'.format(linear_CKA_fast(K_np, K2_np)))
    print(time.time() - start3)
    # X = np.random.randn(150, 2048)
    # gamma = 0.01
    # var = 5.0
    # # start = time.time()
    # # kx1 = var * rbf_np(X, sigma=gamma)
    # # print(time.time() - start) #62sec
    # # print(kx1.shape)
    # start = time.time()
    # kx2 = var * rbf_kernel(X, gamma=gamma)
    # print(time.time() - start) # 11 sec
    # print(kx2.shape)

    # start = time.time()
    # X = torch.from_numpy(X) #.to('cuda')
    # kx3 = var * rbf_torch_kernel(X, X, gamma=gamma)
    # print(time.time() - start) # 6sec se cuda , 7.8 se cpu
    # print(kx3.size)
    # datapath = 'D:/Datasets/data' #'D:/DesktopC/Datasets/data/' 
    # subset = True #True
    # num_classes = 10
    # num_workers = 1
    # batch_size=32
    # seed=2024
    # num_partitions = 10
    # alpha = 0.5
    # partitioning = 'dirichlet' 
    # balance=True
    # # partitioning = 'iid'
    # val_ratio = 0.0
    # trainloaders, validationloaders, testloader, memoryloader, radloader = load_dataset_SSL(datapath, subset, num_classes, num_workers, 
    #                                                            num_partitions, batch_size, partitioning, alpha, balance, seed, rad_ratio=0.02)
    # print()
    pass

# torch.Size([150, 2048])
# Linear CKA, between X and Y: 1.0
# 0.012998819351196289
# Kernel CKA, between X and Y: 1.0
# 0.00512385368347168
# Linear CKA, between X and Y: 0.6546446681022644
# 0.0

# numpy
# Linear CKA, between X and Y: 0.6546446495876963
# 0.007046222686767578
# Kernel CKA, between X and Y: 0.9999999999999998
# 0.007695198059082031
# Linear CKA, between X and Y: 0.6546446495876963
# 0.0
