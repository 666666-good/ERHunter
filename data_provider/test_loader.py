# data_provider/data_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import Dataset, random_split
from config import BATCH_SIZE, STEP, SHUFFLE_FLAG, NUM_WORKERS, DROP_LAST, TEST_BENIGN_DATA_DIR, TEST_MALICIOUS_DATA_DIR, SEQ_LENGTH_CNN
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class TEST_TimeSeriesDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.step = STEP
        self.seq_len = SEQ_LENGTH_CNN
        self.scaler = StandardScaler()
        self.samples = []

        # 预处理所有文件，生成样本索引
        for file_idx, file_path in enumerate(self.files):
            data = np.load(file_path)  # 直接加载npy文件
            data = self.scaler.fit_transform(data)
            #print(file_path)
            #for i in range(0, 2000, self.step):
            for i in range(self.seq_len, len(data), self.step):
            #for i in range(self.seq_len+550, self.seq_len+551):
                # 确保从当前索引 i 往左取 self.seq_len 个元素，不足的用第一个元素补充
                if i >= self.seq_len:
                    sample = data[i - self.seq_len:i]
                    '''
                    grayscale_values = sample.mean(axis=1)  # 每行平均值
                    grayscale_normalized = (grayscale_values - grayscale_values.min()) / ( grayscale_values.max() - grayscale_values.min()) * 255
                    grayscale_normalized = grayscale_normalized.astype(np.uint8)  # 转换为 uint8 类型
                    inverted_grayscale = 255 - grayscale_normalized
                    grayscale_matrix = inverted_grayscale.reshape(40, 20)
                    plt.figure(figsize=(7, 5))
                    #plt.subplots_adjust(left=0.4, right=0.66, top=0.6, bottom=0.4)
                    plt.imshow(grayscale_matrix, cmap='gray')
                    plt.axis('off')  # 不显示坐标轴
                    #plt.title("40x20 Grayscale Image from 16D Data")
                    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
                    image_path = os.path.join(desktop_path, "grayscale_40x20_image.eps")
                    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
                    plt.show()
                    '''
                else:
                    sample = np.concatenate([np.tile(data[0], (self.seq_len - i, 1)), data[:i]], axis=0)
                label = [1, 0] if 'benign' in file_path else [0, 1]
                self.samples.append((sample, label, file_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label, file_path = self.samples[idx]
        sample = torch.tensor(sample, dtype=torch.float32)  # 转为torch tensor
        label = torch.tensor(label, dtype=torch.long)       # 标签也转为long类型
        return sample, label, file_path

def get_test_dataloader(batch_size=BATCH_SIZE, shuffle_flag=SHUFFLE_FLAG, num_workers=NUM_WORKERS, drop_last=DROP_LAST, \
                        benign_data_dir=TEST_BENIGN_DATA_DIR, malicious_data_dir=TEST_MALICIOUS_DATA_DIR):
    benign_data = TEST_TimeSeriesDataset(data_dir=benign_data_dir)
    malicious_data = TEST_TimeSeriesDataset(data_dir=malicious_data_dir)
    print("benign len", len(benign_data))
    print("malicious len", len(malicious_data))
    all_data = ConcatDataset([benign_data, malicious_data])
    #print(len(all_data))
    test_dataloader = DataLoader(
        all_data,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要 shuffle
        num_workers=num_workers,
        drop_last=drop_last
    )
    num_batches = len(test_dataloader)
    #print("num_batches: ", num_batches)
    return test_dataloader, num_batches
