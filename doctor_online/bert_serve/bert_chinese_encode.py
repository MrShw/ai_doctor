import torch
import torch.nn as nn


# source = '/root/.cache/torch/hub/huggingface_pytorch-transformers_main'
source = 'huggingface/pytorch-transformers'
model_name = 'bert-base-chinese'


# model =  torch.hub.load(source, 'model', model_name, source='local')
model = torch.hub.load(source, 'model', model_name, source='github')

# tokenizer = torch.hub.load(source, 'tokenizer', model_name, source='local')
tokenizer = torch.hub.load(source, 'tokenizer', model_name, source='github')


def get_bert_encode(text_1, text_2, mark=102, max_len=10):
    indexed_token = tokenizer.encode(text_1, text_2)
    k = indexed_token.index(mark)

    if len(indexed_token[:k]) >= max_len:
        indexed_token_1 = indexed_token[:max_len]
    else:
        indexed_token_1 = indexed_token[:k] + (max_len-len(indexed_token[:k]))*[0]

    if len(indexed_token[k:]) >= max_len:
        indexed_token_2 = indexed_token[k:k+max_len]
    else:
        indexed_token_2 = indexed_token[k:] + (max_len-len(indexed_token[k:]))*[0]

    indexed_token = indexed_token_1 + indexed_token_2
    segments_ids = [0]*max_len + [1]*max_len

    segments_tensor = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor([indexed_token])

    with torch.no_grad():
        encoded_layers = model(tokens_tensor, token_type_ids=segments_tensor)[0]

    return encoded_layers

if __name__ == '__main__':
    text_1 = "我就试试效果"
    text_2 = "这东西凑合用"
    encoded_layers = get_bert_encode(text_1, text_2)[0]
    print(encoded_layers)
    print(encoded_layers.shape)












