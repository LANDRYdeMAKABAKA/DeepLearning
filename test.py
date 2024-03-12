import torch  # 导入PyTorch库，这是一个用于深度学习的开源库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import numpy as np  # 导入NumPy库，这是一个用于处理数值数据的库
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，这是一个用于绘图的库

# 定义函数
def func(x):  # 定义一个函数func，输入是x
    return np.sin(x)  # 函数的输出是x的正弦值

# 生成训练集和测试集
x_train = np.random.uniform(-np.pi, np.pi, 1000)  # 在[-π, π]区间内随机生成1000个训练数据
y_train = func(x_train)  # 使用定义的函数计算训练数据的标签
x_test = np.linspace(-np.pi, np.pi, 100)  # 在[-π, π]区间内均匀生成100个测试数据
y_test = func(x_test)  # 使用定义的函数计算测试数据的标签

# 定义神经网络模型
class Net(nn.Module):  # 定义一个名为Net的类，继承自nn.Module
    def __init__(self):  # 类的初始化函数
        super(Net, self).__init__()  # 调用父类的初始化函数
        self.fc1 = nn.Linear(1, 50)  # 定义第一层全连接层，输入维度为1，输出维度为50
        self.fc2 = nn.Linear(50, 1)  # 定义第二层全连接层，输入维度为50，输出维度为1
        self.relu = nn.ReLU()  # 定义ReLU激活函数

    def forward(self, x):  # 定义前向传播函数
        x = self.relu(self.fc1(x))  # 输入x经过第一层全连接层和ReLU激活函数
        x = self.fc2(x)  # 然后经过第二层全连接层
        return x  # 返回输出

# 训练模型
model = Net()  # 实例化模型
criterion = nn.MSELoss()  # 定义损失函数为均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 定义优化器为Adam，学习率为0.01

for epoch in range(1000):  # 对整个训练集迭代1000次
    inputs = torch.from_numpy(x_train).float().unsqueeze(1)  # 将训练数据从NumPy数组转换为PyTorch张量，并增加一个维度
    targets = torch.from_numpy(y_train).float().unsqueeze(1)  # 将训练标签从NumPy数组转换为PyTorch张量，并增加一个维度
    outputs = model(inputs)  # 将输入数据传入模型，得到输出
    loss = criterion(outputs, targets)  # 计算输出和目标之间的损失
    optimizer.zero_grad()  # 清零优化器的梯度
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新模型参数

# 验证模型
x = torch.from_numpy(x_test).float().unsqueeze(1)  # 将测试数据从NumPy数组转换为PyTorch张量，并增加一个维度
y = model(x).detach().numpy()  # 将输入数据传入模型，得到输出，并将输出从PyTorch张量转换为NumPy数组
plt.plot(x_test, y_test, label='True function')  # 绘制真实函数
plt.plot(x_test, y, label='Fitted function')  # 绘制拟合函数
plt.legend()  # 显示图例
plt.show()  # 显示图像
