import torch
import torch.nn as nn
import torch.nn.functional as F

def multipcrossentropyLoss(input, target):
    reduction = 'mean'
    ignore_index = -100

    logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

    if ignore_index >= 0:
        notice_index = [i for i in range(target.shape[-1]) if i != ignore_index]
        output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), dim=1)

        if reduction == 'mean':
            return torch.mean(output[target[:, ignore_index] != 1])
        elif reduction == 'sum':
            return torch.sum(output[target[:, ignore_index] != 1])
        else:
            return output[target[:, ignore_index] != 1]
    else:
        output = torch.sum(-target * logsoftmax(input), dim=1)

        if reduction == 'mean':
            return torch.mean(output)
        elif reduction == 'sum':
            return torch.sum(output)
        else:
            return output