# data_provider/data_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from config import BATCH_SIZE, STEP, SHUFFLE_FLAG, NUM_WORKERS, DROP_LAST, BENIGN_DATA_DIR, MALICIOUS_DATA_DIR, NB, L

class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, l=L, s=S, step=STEP, nb=NB, pad_token=0):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.l = l
        self.s = s
        self.step = step
        self.nb = nb  # 每 `nb` 个短历史更新一次长历史
        self.pad_token = pad_token
        self.scaler = StandardScaler()
        self.samples = []  # 用于存储每个短历史的索引和当前的长历史

        # 预处理所有文件，生成样本索引
        for file_idx, file_path in enumerate(self.files):
            data = pd.read_csv(file_path).values
            data = self.scaler.fit_transform(data)
            start_index = L - 1

            for i in range(0, (len(data[start_index:]) - self.s) // self.step + 1 ):
                #print("len",(len(data[start_index:]) - self.s) // self.step)
                if i % nb == 0:
                    # 每 `nb` 个短历史片段更新一次长历史
                    last_index = start_index + i * self.step
                    long_history_start = max(0, last_index - self.l)
                    long_history = data[long_history_start:last_index]
                    if long_history.shape[0] < self.l:
                        oldest_value = long_history[0]
                        padding = np.tile(oldest_value, (self.l - long_history.shape[0], 1))
                        long_history = np.vstack((padding, long_history))
                    self.samples.append((file_idx, i, long_history))
                    #print("i", i)
                else:
                    # 使用之前的长历史
                    self.samples.append((file_idx, i, long_history))
                    #print("i", i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        file_idx, sample_idx, long_history = self.samples[idx]
        data = pd.read_csv(self.files[file_idx]).values
        data = self.scaler.fit_transform(data)
        start_index = L - 1

        # 获取单个短历史
        short_history = data[start_index + sample_idx * self.step: start_index + sample_idx * self.step + self.s]
        #print("start_index + sample_idx * self.step: start_index + sample_idx * self.step + self.s", start_index + sample_idx * self.step, start_index + sample_idx * self.step + self.s)

        if 'benign' in self.files[file_idx]:
            label = [[1, 0]] * self.s
        elif 'malicious' in self.files[file_idx]:
            label = [[0, 1]] * self.s

        # 生成 mask
        mask = [1] * self.l  # 假设全部都有效

        short_history = torch.tensor(
            np.array(short_history), dtype=torch.float32
        ).view(-1, self.s, data.shape[1])

        long_history = torch.tensor(
            np.array(long_history), dtype=torch.float32
        ).view(-1, self.l, data.shape[1])

        mask = torch.tensor(
            np.array(mask), dtype=torch.bool
        ).view(-1, self.l)

        label = torch.tensor(
            np.array(label), dtype=torch.float32
        ).view(-1, self.s, 2)
        #print("long_history, short_history, mask, label", long_history.shape, short_history.shape, mask.shape, label.shape)
        return long_history, short_history, mask, label


def get_test_dataloader(batch_size=BATCH_SIZE, shuffle_flag=SHUFFLE_FLAG, num_workers=NUM_WORKERS, drop_last=DROP_LAST, benign_data_dir=BENIGN_DATA_DIR, malicious_data_dir=MALICIOUS_DATA_DIR):
    benign_data = TimeSeriesDataset(data_dir=benign_data_dir)
    malicious_data = TimeSeriesDataset(data_dir=malicious_data_dir)
    all_data = ConcatDataset([benign_data, malicious_data])

    return DataLoader(
        all_data,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last
    )
