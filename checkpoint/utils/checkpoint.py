import torch
import os

def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    #print(f"Checkpoint saved at {path}")

def load_checkpoint(path, model, optimizer=None):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}, epoch {checkpoint['epoch']}")
        return model, optimizer
    else:
        print(f"No checkpoint found at {path}")
        return 0
