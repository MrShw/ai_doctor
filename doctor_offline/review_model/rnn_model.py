import torch
import torch.nn as nn

"""
input_size: 输入张量最后一个维度的大小
hidden_size: 隐藏层张量最后一个维度的大小
output_size: 输出层张量最后一个维度的大小
"""


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.i2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input1, hidden1):
        combined = torch.cat((input1, hidden1), dim=1)  # 拼接x(t) h(t-1)
        hidden = self.i2h(combined) # 进入第一个Linear
        hidden = self.tanh(hidden)  # 经过激活函数
        output = self.i2o(hidden)   # 进入第二个Linear层
        output = self.softmax(output)   # 进入softmax

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


if __name__ == '__main__':
    input_size = 768    # bert模型输出的维度
    hidden_size = 128   # 自定义的
    n_categories = 2    # 类别数量

    input = torch.rand(1, input_size)   # 随机生成符合形状的张量
    hidden = torch.rand(1, hidden_size)

    rnn = RNN(input_size, hidden_size, n_categories)
    outputs, hidden = rnn(input, hidden)
    print("outputs:", outputs)
    print("hidden:", hidden)
