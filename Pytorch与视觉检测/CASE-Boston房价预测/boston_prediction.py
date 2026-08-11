# -*- coding: utf-8 -*-
"""
波士顿房价预测 - PyTorch神经网络
单隐藏层，10个神经元，20%数据用于验证
"""

import sys
import io

# Windows控制台UTF-8输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置matplotlib中文字体，解决图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框


# 定义神经网络模型：输入层 -> 隐藏层(10神经元) -> 输出层
class BostonNet(nn.Module):
    def __init__(self, input_dim):
        super(BostonNet, self).__init__()
        self.hidden = nn.Linear(input_dim, 10)
        self.output = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x


def main():
    # 1. 加载数据
    df = pd.read_csv('./Pytorch与视觉检测/CASE-Boston房价预测/housing.csv', sep=r'\s+', header=None)

    # 波士顿数据集：前13列为特征，最后一列为目标(房价)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

    # 2. 切分数据集：80%训练，20%验证
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. 特征标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)

    # 4. 转为PyTorch张量
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 5. 构建模型
    input_dim = X_train.shape[1]
    model = BostonNet(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 6. 训练
    epochs = 200
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 每个epoch计算验证集损失
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            val_losses.append(val_loss)
        model.train()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, 训练损失: {avg_train_loss:.6f}, 验证损失: {val_loss:.6f}")

    # 7. 绘制Loss曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='训练损失', color='#2E86AB')
    plt.plot(range(1, epochs + 1), val_losses, label='验证损失', color='#E94F37')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('波士顿房价预测 - Loss曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    plt.show()

    # 8. 验证集评估
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
        val_rmse = np.sqrt(val_loss)

    print(f"\n验证集 MSE: {val_loss:.6f}")
    print(f"验证集 RMSE: {val_rmse:.6f}")

    # 将预测值反标准化后展示
    val_pred_orig = scaler_y.inverse_transform(val_pred.numpy())
    y_val_orig = scaler_y.inverse_transform(y_val_t.numpy())
    print(f"\n前5个样本预测值 vs 真实值:")
    for i in range(min(5, len(val_pred_orig))):
        print(f"  预测: {val_pred_orig[i, 0]:.2f}, 真实: {y_val_orig[i, 0]:.2f}")


if __name__ == '__main__':
    main()
