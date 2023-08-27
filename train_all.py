import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
from torch import nn
from torch.autograd import Variable
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子


setup_seed(19)
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
device = torch.device("cuda")

torch.cuda.is_available()
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# 加载traces
with open("traces.pkl", "rb") as f:
    traces = pickle.load(f)
func1 = "cc5bb2108cc7daf53f9728ad21f661a8ef9c8b36284bacfcb712e2be87eef842"  # 0-1000
func2 = "8e5f533dbf1092f56ac6c7542ef3bdec4661bd442c9b5e7537fabc7b8c03f5a8"  # 0-700
func3 = "762835950e81a11cd04cedcb05275dc111c651625d575077fce49f82170e0986"
func_name = func1
lower_bound = 300
upper_bound = 1350
feature = 10
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
train_size = int((upper_bound - lower_bound) * 0.7) + lower_bound
test_size = upper_bound - train_size
train_X = data_X[:train_size - lower_bound]
train_Y = data_Y[:train_size - lower_bound]
test_X = data_X[train_size - lower_bound:upper_bound - lower_bound]
test_Y = data_Y[train_size - lower_bound:upper_bound - lower_bound]

import torch

# 转换为pytorch张量（自动计算、序列维度、特征维度/目标维度）
train_X = train_X.reshape(-1, 1, feature)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, feature)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)
train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)

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


net = lstm_reg(feature, 4).to(device)  # 实例化
# 定义损失函数、优化器
criterion = nn.MSELoss()  # 均方误差
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)  # 优化器
train_range_l = lower_bound
train_range_r = train_size

# 开始训练
def train(epoch=200):
     net.train()
     for e in range(epoch):
        # 对象转换
        var_x = Variable(train_x).to(device)
        var_y = Variable(train_y).to(device)
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
train(700)
net = net.eval().to(device)  # 转换成测试模式
pred_X = data_X[:train_size - lower_bound]  # 预测前70%
pred_X = pred_X.reshape(-1, 1, feature)
pred_X = torch.from_numpy(pred_X).to(device)  # 转换为张量
var_data = Variable(pred_X)
pred_test = net(var_data)  # 整个数据集的拟合结果
pred_test = pred_test.cpu().view(-1).data.numpy()
print("len", len(data_X))
# 测试集模型动态变化
for i in range(train_size, len(data_X) + lower_bound):
    # 预测下一点
    net = net.eval().to(device)
    single_prediction_input = data_X[i - lower_bound]  # 预测下一点
    # 格式转换
    single_prediction_input = single_prediction_input.reshape(-1, 1, feature)
    single_prediction_input = torch.from_numpy(single_prediction_input)
    var_single_prediction_input = Variable(single_prediction_input).to(device)
    # 预测
    single_prediction_output = net(var_single_prediction_input)
    single_prediction_output = single_prediction_output[0, 0, 0].cpu().detach().numpy()
    # 添加元素
    pred_test = np.append(pred_test, single_prediction_output)
    freq = 10
    if (i - train_size) % freq == 0:
        print( i - train_size, '/', upper_bound - train_size)
        # 重新设置训练集
        train_range_l += freq
        train_range_r += freq
        # 固定窗口
        # train_X = data_X[train_range_l - lower_bound:train_range_r - lower_bound]
        # train_Y = data_Y[train_range_l - lower_bound:train_range_r - lower_bound]
        # 累计训练集
        train_X = data_X[:i - lower_bound]
        train_Y = data_Y[:i - lower_bound]
        train_X = train_X.reshape(-1, 1, feature)
        train_Y = train_Y.reshape(-1, 1, 1)
        train_x = torch.from_numpy(train_X).to(device)
        train_y = torch.from_numpy(train_Y).to(device)
        # 重新训练
        net.train()
        train(200)


# 反归一化预测结果
pred_test_original = pred_test * scalar

# 修正预测结果
for i in range(0, len(pred_test_original)):
    if pred_test_original[i] < 0:
        pred_test_original[i] = 0
for i in range(0, len(pred_test_original)):
    pred_test_original[i] = int(pred_test_original[i])
dataset_original = np.array(dataset) * scalar
# 绘制实际结果和预测结果
x_values = np.arange(feature, upper_bound - lower_bound + 1) + lower_bound
plt.plot(x_values, pred_test_original, "r", label="prediction")
plt.plot(x_values, dataset_original[feature:], "b", label="real")
# 添加竖线标注训练和测试数据的分隔处
plt.axvline(x=train_size, color="g", linestyle="--", label="Train/Test Split")
plt.title(f"{func_name[:5]}(dynamic)", fontsize=12)
plt.xlabel("Time (minutes)")  # 设置 x 轴标签
plt.ylabel("Requests")  # 设置 y 轴标签
plt.legend(loc="best")
plt.savefig("pic.jpg")
