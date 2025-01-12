import torch
def calculate_accuracy(outputs, labels):
    # 初始化计数
    #print(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)  # 获取每个样本的预测类别
    _, true_labels = torch.max(labels, dim=1)  # 获取每个样本的真实标签类别
    #print(predicted, true_labels)
    TP = ((predicted == 1) & (true_labels == 1)).sum().item()
    FP = ((predicted == 1) & (true_labels == 0)).sum().item()
    TN = ((predicted == 0) & (true_labels == 0)).sum().item()
    FN = ((predicted == 0) & (true_labels == 1)).sum().item()
    #print(TP, FP, TN, FN)
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    return accuracy

def calculate_TPFPTNFN(outputs, labels):
    # 初始化计数
    #print(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)  # 获取每个样本的预测类别
    _, true_labels = torch.max(labels, dim=1)  # 获取每个样本的真实标签类别
    #print(predicted, true_labels)
    TP = ((predicted == 1) & (true_labels == 1)).sum().item()
    FP = ((predicted == 1) & (true_labels == 0)).sum().item()
    TN = ((predicted == 0) & (true_labels == 0)).sum().item()
    FN = ((predicted == 0) & (true_labels == 1)).sum().item()
    #print(TP, FP, TN, FN)
    return TP, FP, TN, FN

def calculate_FPR(outputs, labels):
    # 初始化计数
    # print(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)  # 获取每个样本的预测类别
    _, true_labels = torch.max(labels, dim=1)  # 获取每个样本的真实标签类别
    # print(predicted, true_labels)
    TP = ((predicted == 1) & (true_labels == 1)).sum().item()
    FP = ((predicted == 1) & (true_labels == 0)).sum().item()
    TN = ((predicted == 0) & (true_labels == 0)).sum().item()
    FN = ((predicted == 0) & (true_labels == 1)).sum().item()
    # print(TP, FP, TN, FN)
    total = TP + TN + FP + FN
    FPR = FP / (FP+TN) if (FP+TN) > 0 else 0
    return FPR

def calculate_Precision(outputs, labels):
    # 初始化计数
    # print(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)  # 获取每个样本的预测类别
    _, true_labels = torch.max(labels, dim=1)  # 获取每个样本的真实标签类别
    # print(predicted, true_labels)
    TP = ((predicted == 1) & (true_labels == 1)).sum().item()
    FP = ((predicted == 1) & (true_labels == 0)).sum().item()
    TN = ((predicted == 0) & (true_labels == 0)).sum().item()
    FN = ((predicted == 0) & (true_labels == 1)).sum().item()
    # print(TP, FP, TN, FN)
    total = TP + TN + FP + FN
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    return precision

def calculate_Recall(outputs, labels):
    # 初始化计数
    # print(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)  # 获取每个样本的预测类别
    _, true_labels = torch.max(labels, dim=1)  # 获取每个样本的真实标签类别
    # print(predicted, true_labels)
    TP = ((predicted == 1) & (true_labels == 1)).sum().item()
    FP = ((predicted == 1) & (true_labels == 0)).sum().item()
    TN = ((predicted == 0) & (true_labels == 0)).sum().item()
    FN = ((predicted == 0) & (true_labels == 1)).sum().item()
    # print(TP, FP, TN, FN)
    total = TP + TN + FP + FN
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    return recall

    '''
    for i in range(labels.size(0)):
        for j in range(labels.size(1)):
            label = labels[i, j]
            output = outputs[i, j]
            print(label, output)
            # 判断标签和预测结果，根据条件更新计数
            if label[0] == 0 and label[1] == 1:  # 标签为 [0, 1]
                if output[0] < output[1]:  # 预测为 [0, 1]
                    TP = TP + 1
                else:  # 预测为 [1, 0]
                    FN = FN + 1
            elif label[0] == 1 and label[1] == 0:  # 标签为 [1, 0]
                if output[0] > output[1]:  # 预测为 [1, 0]
                    TN = TN + 1
                else:  # 预测为 [0, 1]
                    FP = FP + 1

    # 根据 TP、TN、FP、FN 计算准确率
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    return accuracy
    '''