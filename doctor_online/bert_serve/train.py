import pandas as pd
from sklearn.utils import shuffle
from functools import reduce
from collections import Counter
from bert_chinese_encode import get_bert_encode
import torch
import torch.nn as nn
import time
from finetuning_net import Net
import torch.optim as optim


max_len=10
embedding_size = 768
char_size = 2 * max_len

net = Net(embedding_size, char_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def train(train_data_labels):
    train_running_loss = 0.0
    train_running_acc = 0.0
    for train_tensor, train_labels in train_data_labels:
        optimizer.zero_grad()
        train_outputs = net(train_tensor)
        train_loss = criterion(train_outputs, train_labels)
        train_running_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()
        train_running_acc += (train_outputs.argmax(1) == train_labels).sum().item()
    return train_running_loss, train_running_acc

def valid(valid_data_labels):
    valid_running_loss = 0.0
    valid_running_acc = 0.0
    for valid_tensor, valid_labels in valid_data_labels:
        with torch.no_grad():
            valid_outputs = net(valid_tensor)
            valid_loss = criterion(valid_outputs, valid_labels)
            valid_running_loss += valid_loss.item()
            valid_running_acc += (valid_outputs.argmax(1) == valid_labels).sum().item()

    return valid_running_loss,  valid_running_acc


def data_loader(data_path, batch_size, split=0.2):

    data = pd.read_csv(data_path, header=None, sep='\t')
    print("数据集的正负样本数量:")
    print(dict(Counter(data[0].values)))

    data = shuffle(data).reset_index(drop=True)

    split_point = int(len(data)*split)
    valid_data = data[:split_point]
    train_data = data[split_point:]

    if len(valid_data) < batch_size:
        raise ("Batch size or split not match")

    def _loader_generator(data):

        for batch in range(0, len(data), batch_size):
            batch_encoded = []
            batch_labels = []

            for item in data[batch: batch+batch_size].values.tolist():
                encoded = get_bert_encode(item[1], item[2])[0]
                batch_encoded.append(encoded)

                batch_labels.append([item[0]])

            encoded = reduce(lambda x, y: torch.cat((x, y), dim=0), batch_encoded)
            labels = torch.tensor(reduce(lambda  x, y: x + y, batch_labels))

            yield (encoded, labels)

    return _loader_generator(train_data), _loader_generator(valid_data), len(train_data), len(valid_data)

if __name__ == '__main__':
    # data_path = "./train_data.csv"
    # # 定义batch_size大小
    # batch_size = 32
    #
    # train_data, valid_data, \
    # train_data_len, valid_data_len = data_loader(data_path, batch_size)
    #
    # print(next(train_data))
    # print(next(valid_data))
    # print("train_data_len:", train_data_len)
    # print("valid_data_len:", valid_data_len)

    epochs = 20
    batch_size = 32
    data_path = "./train_data.csv"

    all_train_losses = []
    all_valid_losses = []
    all_train_acc = []
    all_valid_acc = []

    for epoch in range(epochs):
        print("Epoch:", epoch + 1)
        train_data_labels, valid_data_labels, train_data_len, valid_data_len = data_loader(data_path, batch_size)
        train_running_loss, train_running_acc = train(train_data_labels)

        valid_running_loss, valid_running_acc = valid(valid_data_labels)
        train_average_loss = train_running_loss * batch_size / train_data_len
        valid_average_loss = valid_running_loss * batch_size / valid_data_len

        train_average_acc = train_running_acc / train_data_len
        valid_average_acc = valid_running_acc / valid_data_len

        all_train_losses.append(train_average_loss)
        all_valid_losses.append(valid_average_loss)
        all_train_acc.append(train_average_acc)
        all_valid_acc.append(valid_average_acc)

        print("Train Loss:", train_average_loss, "|", "Train Acc:", train_average_acc)
        print("Valid Loss:", valid_average_loss, "|", "Valid Acc:", valid_average_acc)

    print('Finished Training')

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator


    plt.figure(0)
    plt.plot(all_train_losses, label="Train Loss")
    plt.plot(all_valid_losses, color="red", label="Valid Loss")
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(1, epochs)
    plt.legend(loc='upper left')
    # plt.savefig("./loss.png")

    plt.figure(1)
    plt.plot(all_train_acc, label="Train Acc")
    plt.plot(all_valid_acc, color="red", label="Valid Acc")
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(1, epochs)
    plt.legend(loc='upper left')
    # plt.savefig("./acc.png")


    time_ = int(time.time())
    MODEL_PATH = './model/BERT_net_%d.pth' % time_
    torch.save(net.state_dict(), MODEL_PATH)

