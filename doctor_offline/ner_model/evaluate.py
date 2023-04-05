import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from bilstm_crf import NER

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model=None, tokenizer=None, data=None):
    if data is None:
        data = load_from_disk('ner_data/bilstm_crf_data_aidoc')['valid']
    if model is None:
        model_param = torch.load('model/BiLSTM-CRF-final.bin', map_location=torch.device('mps'))
        model = NER(**model_param['init'])
        model.load_state_dict(model_param['state'])
    if tokenizer is None:
        tokenizer = BertTokenizer(vocab_file='ner_data/bilstm_crf_vocab_aidoc.txt')

    total_entities = get_true_entitie(data)
    model_entities = get_pred_entities(data, model, tokenizer)
    indicators = cal_prf(model_entities, total_entities)

    return indicators


def cal_prf(model_entities, total_entities):
    indicators = []

    total_pred_correct = 0
    total_true_correct = 0
    for key in ['DIS', 'SYM']:
        true_entities = total_entities[key]
        true_entities_num = len(true_entities)
        pred_entities = model_entities[key]
        pred_entities_num = len(pred_entities)

        pred_correct = 0  # TP
        pred_incorrect = 0  # FP
        for pred_entity in pred_entities:
            if pred_entity in true_entities:
                pred_correct += 1
                continue
            pred_incorrect += 1

        total_pred_correct += pred_correct
        total_true_correct += true_entities_num

        recall = pred_correct / true_entities_num
        precision = pred_correct / pred_entities_num

        f1 = 0
        if recall != 0 or precision != 0:
            f1 = 2 * recall * precision / (recall + precision)

        print(key, '查全率: %.3f' % recall)
        print(key, '查准率: %.3f' % precision)
        print(key, 'f1: %.3f' % f1)
        print('-' * 50)

        indicators.append([recall, precision, f1])

    print('准确率：%.3f' % (total_pred_correct / total_true_correct))
    indicators.append(total_pred_correct / total_true_correct)

    return indicators



def get_pred_entities(data, model, tokenizer):
    if model is None:
        model_param = torch.load('model/BiLSTM-CRF-final.bin', map_location=torch.device('mps'))
        model = NER(**model_param['init'])
        model.load_state_dict(model_param['state'])
    if tokenizer is None:
        tokenizer = BertTokenizer(vocab_file='ner_data/bilstm_crf_vocab_aidoc.txt')
    model_entities = {'DIS': [], 'SYM': []}

    def start_evaluate(data_inputs):
        model_inputs = tokenizer.encode(data_inputs, return_tensors='pt', add_special_tokens=False)[0]
        model_inputs = model_inputs.to(device)

        with torch.no_grad():
            label_list = model.predict(model_inputs)
        extract_entities = extract_decode(label_list, ''.join(data_inputs.split()))

        nonlocal model_entities

        for key, value in extract_entities.items():
            model_entities[key].extend(value)

    data.map(start_evaluate, input_columns=['data_inputs'], batched=False)

    return model_entities


def get_true_entitie(data):
    total_entities = {'DIS': [], 'SYM': []}

    def calculate_handler(data_inputs, data_labels):
        data_inputs = ''.join(data_inputs.split())
        extract_entities = extract_decode(data_labels, data_inputs)

        nonlocal total_entities
        for key, value in extract_entities.items():
            total_entities[key].extend(value)

    data.map(calculate_handler, input_columns=['data_inputs', 'data_labels'])

    return total_entities


def extract_decode(data_labels, data_inputs):
    label_to_index = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}
    B_DIS, I_DIS = label_to_index['B-dis'], label_to_index['I-dis']
    B_SYM, I_SYM = label_to_index['B-sym'], label_to_index['I-sym']

    def extract_word(start_index, next_label):
        index, entity = start_index+1, [data_inputs[start_index]]
        # entity = [data_inputs[start_index]]

        for index in range(start_index+1, len(data_labels)):
            if data_labels[index] != next_label:
                break
            entity.append(data_inputs[index])

        return index, ''.join(entity)

    extract_entities = {'DIS':[], 'SYM': []}
    index = 0
    next_label = {B_DIS:I_DIS, B_SYM:I_SYM}
    print(next_label)

    word_class = {B_DIS:'DIS', B_SYM:'SYM'}

    while index < len(data_labels):
        label = data_labels[index]
        if label in next_label.keys():
            index, word = extract_word(index, next_label[label])
            extract_entities[word_class[label]].append(word)
            continue

        index += 1

    return extract_entities


if __name__ == '__main__':
    evaluate()


