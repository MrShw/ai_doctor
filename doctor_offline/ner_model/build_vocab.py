import json

def build_vocab():
    chat_to_id = json.load(open('ner_data/char_to_id.json', mode='r', encoding='utf-8'))
    unique_words = list(chat_to_id.keys())[1:-1]
    unique_words.insert(0, '[UNK]')
    unique_words.insert(0, '[PAD]')
    # print(unique_words)

    with open('ner_data/bilstm_crf_vocab_aidoc.txt', 'w') as file:
        for word in unique_words:
            file.write(word+'\n')

if __name__ == '__main__':
    build_vocab()