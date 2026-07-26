import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split

from .utils import get_one_fitness, LargeScenario

def str_to_timestamp(time_str, format="%Y-%m-%d|%H:%M:%S"):
    return time.mktime(time.strptime(time_str[:19], "%Y-%m-%d|%H:%M:%S"))

def filt_folders(folders, start_time, end_time):
    start_timestamp =  str_to_timestamp(start_time)
    end_timestamp =  str_to_timestamp(end_time)
    result = []
    for folder in folders:
        if str_to_timestamp(folder) >= start_timestamp and str_to_timestamp(folder) <= end_timestamp:
            result.append(folder)
    return result

def load_data(root = '../SBT-data/InterFuser', start='2025-02-17|16:00:00', end='2025-02-18|03:50:00'):
    # root = '/home/guannan/Projects/SBT-data/InterFuser'
    folders = os.listdir(root)
    folders.sort()
    folders = filt_folders(folders, start, end)

    fitnesses = []
    scenarios = []

    for folder in folders:
        path = "{}/{}/".format(root, folder)
        # print(path)
        fitnesses.append(get_one_fitness(path+'criterion.csv', path+'fitness.csv', col='CollisionTest'))
        scenarios.append(LargeScenario(path))
    
    X = np.array([])
    for scenario in scenarios:
        if len(X) == 0:
            X = scenario.data
        else:
            X = np.concatenate([X, scenario.data], axis=0)

    y = np.array([])
    for fitnesse in fitnesses:
        if len(y) == 0:
            y = fitnesse
        else:
            y = np.concatenate([y, fitnesse], axis=0)
    return X, y


# 定义小型神经网络
class SmallNet(nn.Module):
    def __init__(self, input_dim):
        super(SmallNet, self).__init__()
        # self.fc_64   = nn.Linear(input_dim, 64) # 第一层，全连接
        # self.relu    = nn.ReLU()                # ReLU 激活函数
        # self.dropout = nn.Dropout(0.1)          # Dropout 防止过拟合
        # self.fc_32   = nn.Linear(64, 32)        # 第二层，全连接
        # self.fc_out  = nn.Linear(32, 1)         # 输出层，单个回归值
        self.fc_64   = nn.Linear(input_dim, 128)  # 第一层，全连接
        self.relu    = nn.ReLU()                  # ReLU 激活函数
        self.dropout = nn.Dropout(0.1)            # Dropout 防止过拟合
        self.fc_32   = nn.Linear(128, 64)         # 第二层，全连接
        self.fc_out  = nn.Linear(64, 1)           # 输出层，单个回归值


    def forward(self, x):
        x = self.fc_64(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc_32(x)
        x = self.relu(x)

        x = self.fc_out(x)
        return torch.sigmoid(x)  # 输出值在 0-1 范围内

# 自定义加权损失函数
def weighted_mse_loss(output, target):
    # y=0 的样本权重更高
    weights = torch.where(target == 0, 5.0, 1.0)
    # weights = torch.where(target < 0.1, (1-target)*10, 1.0)
    # weights = torch.where(target < 0.5, 1-target, target)
    # weights = torch.where(target < 0.1, 1.0, 1.0)
    loss = weights * (output - target) ** 2
    return loss.mean()

def pairwise_rank_loss(output, target):
    # 计算目标值和预测值的差
    diff_target = target - target.T  # Target 的差
    diff_output = output - output.T  # Output 的差

    # 计算排序损失：如果 diff_target 和 diff_output 符号不一致，则 penalize
    loss = torch.mean(torch.relu(-diff_target * diff_output))  # relu 确保非负
    return loss

def combined_loss(output, target):
    weights = [10,0]
    return (weights[0]*weighted_mse_loss(output, target) + weights[1]*pairwise_rank_loss(output, target))/sum(weights)


def train(root = '/home/guannan/Projects/SBT-data/InterFuser', start='2025-02-17|16:00:00', end='2025-02-18|03:50:00'):
    X, y = load_data(root, start, end)
    # 数据集（占位）
    # 使用你的数据，替换 X 和 y
    # X: shape = (num_samples, 71), y: shape = (num_samples,)
    X = X.astype(np.float32)  # 随机生成特征
    y = y.astype(np.float32)  # 随机生成标签

    # 转换为 PyTorch 张量
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y).view(-1, 1)

    # 超参数
    input_dim = X.shape[1]
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.001

    # K 折交叉验证
    kf = KFold(n_splits=5)
    fold_mse = []


    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.33, random_state=42)
    # for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # print(f"Fold {fold + 1}")
        
        # # 划分训练集和验证集
        # X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        # y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
        
    # 数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    model = SmallNet(input_dim=input_dim)
    # criterion = weighted_mse_loss
    # criterion = pairwise_rank_loss
    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    best_val_loss = float('inf')
    best_val_mse_loss = float('inf')
    early_stop_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证模型
        model.eval()
        val_loss = 0.0
        val_mse_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_mse_loss += weighted_mse_loss(outputs, y_batch).item()
        
        val_loss /= len(val_loader)
        val_mse_loss /= len(val_loader)
        # print(f"Epoch {epoch + 1}: Train Loss = {train_loss / len(train_loader):.4f}, Val Loss = {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mse_loss = val_mse_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                # print("Early stopping triggered!")
                break
    
    # 保存每折的验证损失
    # fold_mse.append(best_val_loss)
    fold_mse.append(best_val_mse_loss)

    # 打印结果
    print("MSE:", np.mean(fold_mse))
    print("RMSE:", np.mean(fold_mse)**0.5)
    print(np.concatenate([np.array(outputs), np.array(y_batch)], axis=1))

    return model