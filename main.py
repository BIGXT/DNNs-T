import file
import torchsummary
from torch.autograd import Variable

if __name__ == '__main__':
    import torch
    from torch import nn
    import numpy as np
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = file.readcsv('X.csv')
    y = file.readcsv('Y.csv')

    # X, y = torch.tensor(X.values), torch.tensor(y.values)
    '''
    with open("data.csv", "r", encoding="utf-8") as f:
        data = f.read()
    data = [row.split(',') for row in data.split("\n")]

    value = [int(each[1]) for each in data]
    '''

    data = X.values.tolist()
    value = y.values.tolist()

    li_x = []
    li_y = []
    seq = 2
    # 因为数据集较少，序列长度太长会影响结果
    for i in range(len(data) - seq):
        li_x.append(value[i: i + seq])
        li_y.append(value[i + seq])

    # 分训练和测试集
    train_x = (torch.tensor(li_x[:-30]).float() / 1000.).reshape(-1, seq, 17).to(device)
    train_y = (torch.tensor(li_y[:-30]).float() / 1000.).reshape(-1, 17).to(device)

    test_x = (torch.tensor(li_x[-30:]).float() / 1000.).reshape(-1, seq, 17).to(device)
    test_y = (torch.tensor(li_y[-30:]).float() / 1000.).reshape(-1, 17).to(device)


    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            input_size = 17
            hidden_size = 32
            num_layers = 3
            batch_first = True
            dropout = 0 #0.5
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=batch_first, dropout=dropout)
            # 输入格式是17,表示17个特征，输出隐藏层大小是32，对于序列比较短的数据num_layers不要设置大，否则效果会变差
            # 原来的输入格式是：(seq, batch, shape)，设置batch_first=True以后，输入格式就可以改为：(batch, seq, shape)，更符合平常使用的习惯
            self.linear = nn.Linear(hidden_size * seq, input_size)

        def forward(self, x):
            x, (h, c) = self.lstm(x)
            x = x.reshape(-1, 32 * seq)
            x = self.linear(x)
            return x


    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fun = nn.MSELoss()

    model.train()
    for epoch in range(300):
        output = model(train_x)

        loss = loss_fun(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 and epoch > 0:
            test_loss = loss_fun(model(test_x), test_y)
            print("epoch:{}, loss:{}, test_loss: {}".format(epoch, loss, test_loss))

    model.eval()

    result = li_x[0][:seq - 1] + list(((model(train_x).data.reshape(-1, 17))* 1000).tolist() ) + list(
        ((model(test_x).data.reshape(-1, 17))* 1000).tolist() )
    # 通过模型计算预测结果并解码后保存到列表里，因为预测是从第seq个开始的，所有前面要加seq-1条数据
    # print(result)

    # 网络参数可视化
    # params = model.state_dict()

    torchsummary.summary(model, (1000, 17))

    # print(model(test_x))
    '''

    plt.plot(value, label="real")
    # 原来的走势
    plt.plot(result, label="pred")
    # 模型预测的走势
    plt.legend(loc='best')
    plt.show()
    '''
