import os
import torch
import torch.nn as nn

from rnn_model import RNN
from bert_chinese_encode import get_bert_encode_for_single

# 超参数
MODEL_PATH = './BERT_RNN.pth'
n_hidden = 128
input_size = 768
n_categories = 2

rnn = RNN(input_size, n_hidden, n_categories)
rnn.load_state_dict(torch.load(MODEL_PATH))


def _test(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[1]):
        output, hidden = rnn(line_tensor[0][i].unsqueeze(0), hidden)

    return output


def predict(text_tensor):
    # 不自动求梯度
    with torch.no_grad():
        output = _test(get_bert_encode_for_single(text_tensor))
        _, topi = output.topk(1, 1)

        return topi.item()


def batch_predict(input_path_noreview, output_path_reviewed):
    csv_list = os.listdir(input_path_noreview)

    for csv in csv_list:
        # 以读的方式，打开需要审核的每一个csv文件
        with open(os.path.join(input_path_noreview, csv), 'r') as fr:
            # 以写的方式，打开输出路径的同名csv文件
            with open(os.path.join(output_path_reviewed, csv), 'w') as fw:
                input_lines = fr.readlines()
                for input_line in input_lines:
                    print(csv, input_line)
                    res = predict(input_line)

                    # 结果是1，把文本写入到文件中
                    if res:
                        fw.write(input_line+'\n')
                    # 丢弃
                    else:
                        pass


if __name__ == '__main__':
    input_path = '../structured/noreview/'
    output_path = '../structured/reviewed/'
    batch_predict(input_path, output_path)




