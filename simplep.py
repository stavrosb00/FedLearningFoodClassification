import torch
import math
import time

def centering(K):
    n = K.size(0)
    unit = torch.ones(n, n, device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n

    return torch.matmul(torch.matmul(H, K), H)


def rbf(X, sigma=None):
    GX = torch.matmul(X, X.t())
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).t()
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = torch.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = torch.matmul(X, X.t())
    L_Y = torch.matmul(Y, Y.t())
    return torch.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


if __name__=='__main__':
    X = torch.randn(25000, 512)
    Y = torch.randn(25000, 512)
    start =time.time()
    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print(time.time() - start)
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

    start =time.time()
    print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
    print(time.time() - start)
    print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))