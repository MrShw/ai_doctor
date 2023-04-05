from flask import Flask
from flask import request
app = Flask(__name__)

import torch
from bert_chinese_encode import get_bert_encode
from finetuning_net import Net

MODEL_PATH = "./model/BERT_net.pth"

embedding_size = 768
char_size = 20
dropout = 0.2

net = Net(embedding_size=embedding_size, char_size=char_size, dropout=dropout)
net.load_state_dict(torch.load(MODEL_PATH))
net.eval()

@app.route('/main_serve/', methods = ["POST"])
def recognition():
    text1 = request.form['text1']
    text2 = request.form['text2']
    print("recognition:", text1, text2)

    inputs = get_bert_encode(text1, text2, mark=102, max_len=10)
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    return str(predicted.item())















