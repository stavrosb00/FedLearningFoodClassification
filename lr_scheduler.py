import torch
import numpy as np
import matplotlib.pyplot as plt
# import math
# from torch.optim.lr_scheduler import _LRScheduler
class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        
        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr

def plot_lr_scheduler(scheduler: LR_Scheduler, num_epochs: int, epochs: int, batches: int):
    lrs = []
    for epoch in range(num_epochs):
        for batch in range(batches):
            lr = scheduler.step()
            lrs.append(lr)
    plt.plot(lrs)
    plt.xlabel('Internal step')
    plt.ylabel('LR')
    plt.title(f'Cosine LR scheduler w/ warm up E={num_epochs}, Batches={batches}')
    plt.savefig('images/lr_scheduler.png')
    plt.show()



if __name__ == "__main__":
    import torchvision
    import time
    model = torchvision.models.resnet50()
    # optimizer = torch.optim.SGD(model.parameters(), lr=999)
    # epochs = 100
    n_iter = 1000
    local_epochs = 200
    base_lr = 0.032 
    batch_size = 128
    init_lr = base_lr*batch_size/256
    warmup_epochs = 10
    warmup_lr = 0
    batches = 59
    num_epochs = 800
    n_iter = batches # 59* 128 ~= 7500 images subset 10 classes (Batches * bs)
    final_lr = 0
    # print(epochs)
    print(n_iter)
    # scheduler = LR_Scheduler(optimizer, 10, 1, epochs, 3, 0, n_iter)
    # start = time.time()
    optimizer = torch.optim.SGD(model.parameters(), lr=999)
    # scheduler = LR_Scheduler(optimizer, warmup_epochs, 0, local_epochs, init_lr, 0, n_iter)
    scheduler = LR_Scheduler(optimizer=optimizer, warmup_epochs=warmup_epochs, warmup_lr=warmup_lr * batch_size / 256, 
                                num_epochs=num_epochs, base_lr=init_lr, final_lr=final_lr * batch_size / 256, iter_per_epoch=n_iter,
                                constant_predictor_lr=False)
    # print(time.time() - start) 0.00100
    # exit()
    
    plot_lr_scheduler(scheduler, num_epochs=local_epochs, epochs=local_epochs, batches=n_iter)
    # lrs = []
    # for epoch in range(num_epochs):
    # # for epoch in range(local_epochs):
    #     for batch in range(batches):
    #         lr = scheduler.step()
    #         lrs.append(lr)
    # plt.plot(lrs)
    # plt.xlabel('Internal step')
    # plt.ylabel('LR')
    # plt.title(f'Cosine LR scheduler w/ warm up E={num_epochs}, Batches={batches}')
    # plt.savefig('/images/lr_scheduler.png')
    # plt.show()