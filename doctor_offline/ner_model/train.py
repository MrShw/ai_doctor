import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from evaluate import evaluate

from bilstm_crf import NER

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pad_batch_inputs(data, labels, tokenizer):
    data_inputs, data_length, data_labels = [], [], []
    for data_input, data_label in zip(data, labels):
        data_input_encode = tokenizer.encode(data_input,
                                             return_tensors='pt',
                                             add_special_tokens=False)
        data_input_encode = data_input_encode.to(device)
        data_inputs.append(data_input_encode.squeeze())
        data_input = ''.join(data_input.split())
        data_length.append(len(data_input))
        data_labels.append(torch.tensor(data_label, device=device))

    sorted_index = np.argsort(-np.asarray(data_length))
    sorted_inputs, sorted_labels, sorted_length = [], [], []

    for index in sorted_index:
        sorted_inputs.append(data_inputs[index])
        sorted_labels.append(data_labels[index])
        sorted_length.append(data_length[index])

    print(sorted_inputs, )
    pad_inputs = pad_sequence(sorted_inputs)
    print(pad_inputs)

    return pad_inputs, sorted_labels, sorted_length


label_to_index = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}


def train():
    train_data = load_from_disk('ner_data/bilstm_crf_data_aidoc')['train']
    tokenizer = BertTokenizer(vocab_file='ner_data/bilstm_crf_vocab_aidoc.txt')
    model = NER(vocab_size=tokenizer.vocab_size, label_num=len(label_to_index))#.cuda(device)

    batch_size = 16
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    num_epoch = 10

    train_history_list = []
    valid_history_list = []

    def start_train(data_inputs, data_labels, tokenizer):
        pad_inputs, sorted_labels, sorted_length = pad_batch_inputs(data_inputs, data_labels, tokenizer)
        loss = model(pad_inputs, sorted_labels, sorted_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        nonlocal total_loss
        total_loss += loss.item()


    for epoch in range(num_epoch):
        total_loss = 0.0
        train_data.map(start_train, input_columns=['data_inputs', 'data_labels'],
                       batched = True,
                       batch_size=batch_size,
                       fn_kwargs={'tokenizer': tokenizer},
                       desc='epoch: %d' % (epoch + 1))
        print('epoch: %d loss: %.3f' % (epoch + 1, total_loss))

        # train_eval_result = evaluate(model, tokenizer, train_data)
        # train_eval_result.append(total_loss)
        # train_history_list.append(train_eval_result)

        model.save_model('model/BiLSTM-CRF-%d.bin' % (epoch + 1))

if __name__ == '__main__':
    train()