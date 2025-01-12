# data_provider/data_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
import random
import pickle
from datetime import datetime
import numpy as np
from torch.utils.data import Dataset, random_split
from config import BATCH_SIZE, STEP, SHUFFLE_FLAG, NUM_WORKERS, DROP_LAST, BENIGN_DATA_DIR, MALICIOUS_DATA_DIR, SEQ_LENGTH_CNN

class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.step = STEP
        self.seq_len = SEQ_LENGTH_CNN
        self.scaler = StandardScaler()
        self.samples = []

        # 预处理所有文件，生成样本索引
        for file_idx, file_path in enumerate(self.files):
            #data = pd.read_csv(file_path).values
            #data = self.scaler.fit_transform(data)  # 对每个文件进行标准化
            data = np.load(file_path)  # 直接加载npy文件
            data = self.scaler.fit_transform(data)
            # 分段数据，并按步长进行处理
            for i in range(0, len(data) - self.seq_len + 1, self.step):
                sample = data[i:i + self.seq_len]  # 取一个步长的窗口，长度为seq_length
                label = [1, 0] if 'benign' in file_path else [0,1]  # 标记恶意和正常样本
                #print(label)
                self.samples.append((sample, label, file_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label, file_path = self.samples[idx]
        #print(sample, label)
        sample = torch.tensor(sample, dtype=torch.float32)  # 转为torch tensor
        label = torch.tensor(label, dtype=torch.long)       # 标签也转为long类型
        return sample, label, file_path

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample, label, file_path = self.data_list[idx]
        sample = sample.clone().detach().float()  # 确保是float32类型
        label = label.clone().detach().long()
        return sample, label, file_path

def get_dataloader(batch_size=BATCH_SIZE, shuffle_flag=SHUFFLE_FLAG, num_workers=NUM_WORKERS, drop_last=DROP_LAST, benign_data_dir=BENIGN_DATA_DIR, malicious_data_dir=MALICIOUS_DATA_DIR):
    benign_data = TimeSeriesDataset(data_dir=benign_data_dir)
    malicious_data = TimeSeriesDataset(data_dir=malicious_data_dir)
    print("benign len", len(benign_data))
    print("malicious len", len(malicious_data))
    all_data = ConcatDataset([benign_data, malicious_data])
    all_data_list = list(all_data)
    total_size = len(all_data_list)
    for i in range(total_size - 1, 0, -1):
        j = random.randint(0, i)  # 随机选择一个索引 j
        all_data_list[i], all_data_list[j] = all_data_list[j], all_data_list[i]
    #for i, (sample, label) in enumerate(all_data_list[:5]):
    #    print(f"Sample {i + 1}: {sample}, Label: {label}")
    shuffled_dataset = CustomDataset(all_data_list)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    #print(train_size, val_size, test_size)

    train_dataset, val_dataset, test_dataset = random_split(
        shuffled_dataset, [train_size, val_size, test_size]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,  # 训练集需要打乱
        num_workers=num_workers,
        drop_last=drop_last
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要 shuffle
        num_workers=num_workers,
        drop_last=drop_last
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要 shuffle
        num_workers=num_workers,
        drop_last=drop_last
    )

    filename = f"dataset_{datetime.today().strftime('%Y-%m-%d')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump({
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        }, f)
    print(f"Dataset saved as {filename}")

    num_batches = len(train_dataloader)
    return train_dataloader, val_dataloader, test_dataloader, num_batches