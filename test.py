import torch
import os
from data_provider.test_loader import get_test_dataloader
from model.transformer_model import ERHunter
from utils.utils import get_loss, get_optimizer
from utils.metrics import calculate_accuracy, calculate_FPR, calculate_Precision, calculate_Recall, calculate_TPFPTNFN
from utils.checkpoint import load_checkpoint
from config import CHECKPOINT_PATH, LR, NUM_CLASSES, BATCH_SIZE, CHECKPOINT_EPOCH
import torch.nn as nn
from collections import defaultdict
import psutil
import time
import threading
from tqdm import tqdm

# 用于记录 CPU 使用率的列表
cpu_percentages = []
monitoring = True

# 定义函数，用于监控 CPU 使用率
def monitor_cpu(interval=0.5):
    while monitoring:
        # 获取 CPU 使用率，并添加到列表中
        cpu_percentages.append(psutil.cpu_percent(interval=None))
        time.sleep(interval)


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data, max_iters = get_test_dataloader()
    model = ERHunter().to(device)
    optimizer = get_optimizer(model, LR)
    criterion = get_loss()
    softmax = nn.Softmax(dim=1)

    model.eval()
    checkpoint_filename = os.path.join(CHECKPOINT_PATH, f"checkpoint-{CHECKPOINT_EPOCH}.pth")
    print(f"Loading checkpoint from {checkpoint_filename}...")

    if os.path.exists(checkpoint_filename):
        load_checkpoint(checkpoint_filename, model, optimizer)
    else:
        print("No checkpoint found, cannot proceed with testing.")
        return

    '''
    monitoring = True  # 开启监控标志
    monitor_thread = threading.Thread(target=monitor_cpu, args=(0.5,))
    monitor_thread.start()
    start_time = time.time()
    '''
    print("Evaluating model on test data...")
    test_loss, test_accuracy = 0, 0
    total_FPR, total_recall, total_precision = 0, 0, 0
    TP, FP, TN, FN = 0, 0, 0, 0
    statistics = defaultdict(list)
    with torch.no_grad():  # 禁用梯度计算以节省内存
        #progress_bar = tqdm(test_data, unit="batch")
        for sample, label, file_path in test_data:
            #print(sample.shape)
            sample, label = (sample.to(device), label.to(device))
            outputs = model(sample)
            #print(outputs.shape)
            loss = criterion(outputs, torch.argmax(label, dim=1))
            test_loss += loss.item()
            probabilities = softmax(outputs)
            '''
            end_time = time.time()
            monitoring = False
            monitor_thread.join()
            avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
            print(f"Program run time: {end_time - start_time:.2f} seconds")
            print(f"Average CPU usage: {avg_cpu:.2f}%")
            '''
            test_accuracy += calculate_accuracy(probabilities, label)
            total_FPR += calculate_FPR(probabilities, label)
            total_recall += calculate_Recall(probabilities, label)
            total_precision += calculate_Precision(probabilities, label)
            TP1, FP1, TN1, FN1 = calculate_TPFPTNFN(probabilities, label)
            TP += TP1
            FP += FP1
            TN += TN1
            FN += FN1
            _, predicted = torch.max(probabilities, dim=1)  # 获取每个样本的预测类别
            _, true_labels = torch.max(label, dim=1)  # 获取每个样本的真实标签类别

            for i in range(len(file_path)):
                true_label = true_labels[i].item()
                predicted_label = predicted[i].item()
                # 统计每个 file_path 对应的 probabilities 和 label
                statistics[file_path[i]].append({
                    'probability-1': probabilities[i][0],
                    'probability-2': probabilities[i][1],
                    'true_label': true_label,
                    'predicted_label': predicted_label
                })
        print(f"Test Accuracy: {test_accuracy / len(test_data):.3f} , Test FPR: {total_FPR / len(test_data):.3f}, Test recall: \
        {total_recall / len(test_data):.3f}, Test precision: {total_precision / len(test_data):.3f}, TP: {TP:.3f}, FP: {FP:.3f}, TN: {TN:.3f}, FN: {FN:.3f}")
        with open(r'C:\Users\Administrator\Desktop\1-test.txt', 'w') as f:
            for file, samples in statistics.items():
                f.write(f"File: {file}\n")
                for sample in samples:
                    f.write(
                        f" probability-1: {sample['probability-1']}, probability-2: {sample['probability-2']}, True Label: {sample['true_label']}, Predicted Label: {sample['predicted_label']}\n")
                f.write("\n")
        result_stats = []
        for file, samples in statistics.items():
            TP = FP = TN = FN = 0
            for sample in samples:
                predicted_label = sample['predicted_label']
                true_label = sample['true_label']
                if predicted_label == 1 and true_label == 1:
                    TP += 1
                elif predicted_label == 1 and true_label == 0:
                    FP += 1
                elif predicted_label == 0 and true_label == 0:
                    TN += 1
                elif predicted_label == 0 and true_label == 1:
                    FN += 1
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            result_stats.append({
                'file_path': file,
                'TP': TP,
                'FP': FP,
                'TN': TN,
                'FN': FN,
                'Accuracy': accuracy
            })
        with open(r'C:\Users\Administrator\Desktop\2-test.txt', 'w') as f:
            for stat in result_stats:
                f.write(f"File: {stat['file_path']}\n")
                f.write(
                    f"TP: {stat['TP']}, FP: {stat['FP']}, TN: {stat['TN']}, FN: {stat['FN']}, Accuracy: {stat['Accuracy']}\n\n")

if __name__ == "__main__":
    test()
