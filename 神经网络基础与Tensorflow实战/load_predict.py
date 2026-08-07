# 加载已保存的 Boston 房价模型，使用数据集中任意一条数据进行预测
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# 加载模型
model = tf.keras.models.load_model('boston_model.keras')

# 加载训练时使用的标准化器
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 读取数据集
data = pd.read_csv('housing.csv', header=None, delim_whitespace=True)
X = data.iloc[:, :-1].values   # 前13列为特征
y = data.iloc[:, -1].values   # 最后一列为真实房价

# 取任意一条数据（这里取第 0 条）
sample_idx = 0
x_raw = X[sample_idx:sample_idx + 1]   # shape (1, 13)
y_real = y[sample_idx]

# 使用保存的标准化器对输入做相同预处理
x_scaled = scaler.transform(x_raw)

# 预测
y_pred = model.predict(x_scaled)

# 输出结果
print('使用的样本索引:', sample_idx)
print('输入特征（原始）:', x_raw.flatten())
print('真实房价:', y_real)
print('预测房价:', y_pred[0][0])
