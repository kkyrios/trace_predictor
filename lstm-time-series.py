import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# 加载traces
with open("traces.pkl", "rb") as f:
    traces = pickle.load(f)
func1 = "cc5bb2108cc7daf53f9728ad21f661a8ef9c8b36284bacfcb712e2be87eef842"  # 0-1000
func2 = "8e5f533dbf1092f56ac6c7542ef3bdec4661bd442c9b5e7537fabc7b8c03f5a8"  # 0-700
func3 = "762835950e81a11cd04cedcb05275dc111c651625d575077fce49f82170e0986"  #
func_name = func1
lower_bound = 0
upper_bound = 1000
feature = 15
request_per_minutes = {}
# 获取特定函数调用的跟踪数据
trace = traces[func_name]

# 遍历函数调用的结束时间列表，统计每分钟的请求数量
for t in range(lower_bound, upper_bound + 1):
    request_per_minutes[t] = 0

for timestamp in trace:
    minute = int(timestamp) // 60
    if lower_bound <= minute <= upper_bound:
        request_per_minutes[minute] += 1

dataset = []
for t in range(lower_bound, upper_bound + 1):
    dataset.append([request_per_minutes[t]])

# 数据预处理(?)
dataset = np.array(dataset)
# print(dataset)
dataset = dataset.astype("float32")  # 转浮点数
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value  # 计算差值用于归一化
dataset = list(map(lambda x: x / scalar, dataset))  # 归一化

# 窗口大小


# look_back：窗口大小，生成输入数据X、目标数据Y
def create_dataset(dataset, look_back=feature):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i : (i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# 创建好输入输出
data_X, data_Y = create_dataset(dataset)
# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

import torch

# 转换为pytorch张量（自动计算、序列维度、特征维度/目标维度）
train_X = train_X.reshape(-1, 1, feature)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, feature)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)

from torch import nn
from torch.autograd import Variable


# 定义模型，构建神经网络(hidden_size、num_layers可调）
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()  # 调用父类构造函数

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # （输入特征，隐藏层，lstm层数）
        self.reg = nn.Linear(hidden_size, output_size)  # 回归预测

    # 向前传播
    def forward(self, x):
        x, _ = self.rnn(x)  # x:lstm层输出结果
        # (seq, batch, hidden)/（序列长，批次大小，隐藏层结果）
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        x = self.reg(x)  # 重塑数据进行回归预测
        x = x.view(s, b, -1)  # 再次重塑
        return x  # 模型预测结果


net = lstm_reg(feature, 4)  # 实例化
# 定义损失函数、优化器
criterion = nn.MSELoss()  # 均方误差
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)  # 优化器


# 开始训练
def train(epoch=800):
    for e in range(epoch):
        # 对象转换
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        # 前向传播
        out = net(var_x)  # out：预测输出
        loss = criterion(out, var_y)  # 误差损失
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e + 1) % 100 == 0:  # 每 100 次输出结果
            print("Epoch: {}, Loss: {:.5f}".format(e + 1, loss.item()))  # (训练周期，损失值)


# 前70%使用原模型
train()
net = net.eval()  # 转换成测试模式
pred_X = data_X  # 预测前70%
pred_X = pred_X.reshape(-1, 1, feature)
pred_X = torch.from_numpy(pred_X)  # 转换为张量
var_data = Variable(pred_X)
pred_test = net(var_data)  # 整个数据集的拟合结果
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()
# 反归一化预测结果
pred_test_original = pred_test * scalar
dataset_original = np.array(dataset) * scalar
# 绘制实际结果和预测结果
plt.plot(pred_test_original, "r", label="prediction")
plt.plot(dataset_original[feature:], "b", label="real")
# 添加竖线标注训练和测试数据的分隔处
plt.axvline(x=train_size, color="g", linestyle="--", label="Train/Test Split")
plt.title(f"{func_name[:5]}", fontsize=12)
plt.legend(loc="best")
plt.show()
