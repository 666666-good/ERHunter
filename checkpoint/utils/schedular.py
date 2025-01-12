import torch
import torch.optim as optim
import torch.nn as nn
from config import EPOCH, CHECKPOINT_PATH, LR, NUM_CLASSES, BATCH_SIZE, PHAZE, WARMUP_EPOCHS, MAX_LR, STEP_SIZE, GAMMA

class WarmUpScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.warmup_epochs = WARMUP_EPOCHS
        self.max_lr = MAX_LR
        self.step_size = STEP_SIZE
        self.gamma = GAMMA
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = self.step_size, gamma = self.gamma)
        self.last_epoch = -1  # 初始epoch值为-1

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 在warmup阶段，逐步增加学习率
            lr = self.max_lr * (epoch / self.warmup_epochs)  # 从0逐步增加到max_lr
            #print("lr",epoch, lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # 在warmup结束后，使用正常的学习率调度器
            self.scheduler.step()
            # 获取调度器更新后的学习率并应用
            lr = self.scheduler.get_last_lr()[0]
            #print("lr",epoch, lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
