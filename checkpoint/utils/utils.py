# utils/utils.py
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Loss function with masking support
def get_loss():
    return nn.CrossEntropyLoss()

# Optimizer
def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)

# Masked loss calculation for handling padding in sequences
def masked_loss(output, target, mask):
    loss_fn = get_loss()
    target = target * mask  # Apply mask to target
    return loss_fn(output[mask], target[mask])
