import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class CRF(nn.Module):
    def __init__(self, label_num):
        super(CRF, self).__init__()
        self.label_num  = label_num
        self.transition_scores = nn.Parameter(torch.randn(self.label_num+2, self.label_num+2))

        self.START_TAG, self.END_TAG = self.label_num, self.label_num+1
        self.fill_value = -1000
        self.transition_scores.data[:, self.START_TAG] = self.fill_value
        self.transition_scores.data[self.END_TAG, :] = self.fill_value

    def _get_real_path_score(self, emission_score, sequence_label):
        seq_len = len(sequence_label)
        real_emission_score = torch.sum(emission_score[list(range(seq_len)), sequence_label])
        b_id = torch.tensor([self.START_TAG], dtype=torch.int32, device=device)
        e_id = torch.tensor([self.END_TAG], dtype=torch.int32,  device=device)
        sequence_label_expand = torch.cat([b_id, sequence_label, e_id])

        pre_tag = sequence_label_expand[list(range(seq_len+1))]
        now_tag = sequence_label_expand[list(range(1, seq_len+2))]
        real_transition_score = torch.sum(self.transition_scores[
            pre_tag, now_tag])

        real_path_score = real_emission_score + real_transition_score

        return real_path_score

    def _log_sum_exp(self, score):
        max_score, _ = torch.max(score, dim=0)
        max_score_expand = max_score.expand(score.shape)
        return max_score + torch.log(torch.sum(torch.exp(score-max_score_expand)))

    def _expand_emission_matrix(self, emission_score):
        seq_length = emission_score.shape[0]
        b_s = torch.tensor([[self.fill_value] * self.label_num + [0, self.fill_value]],
                           device=device)
        e_s = torch.tensor([[self.fill_value] * self.label_num + [self.fill_value, 0]],
                           device=device)

        expand_matrix = self.fill_value * torch.ones([seq_length, 2], dtype=torch.float32,
                                                     device=device)
        emission_score_expand = torch.cat([emission_score, expand_matrix], dim=1)
        emission_score_expand = torch.cat([b_s, emission_score_expand, e_s], dim=0)

        return emission_score_expand

    def _get_total_path_score(self, emission_score):
        emission_score_expand = self._expand_emission_matrix(emission_score)
        pre = emission_score_expand[0]
        for obs in emission_score_expand[1:]:
            pre_expand = pre.reshape(-1, 1).expand([self.label_num+2, self.label_num+2])
            obs_expand = obs.expand([self.label_num+2, self.label_num+2])
            score = obs_expand + pre_expand + self.transition_scores

            # print('\nscore:', score)
            # print('\nscore.shape:', score.shape)
            pre = self._log_sum_exp(score)

        return self._log_sum_exp(pre)

    def forward(self, emission_scores, sequence_labels):
        total = 0.0
        for emission_score, sequence_label in zip(emission_scores, sequence_labels):
            real_path_score = self._get_real_path_score(emission_score, sequence_label)
            total_path_score = self._get_total_path_score(emission_score)
            loss = total_path_score - real_path_score
            total += loss

        return total

    def predict(self, emission_score):
        emission_score_expand = self._expand_emission_matrix(emission_score)
        ids = torch.zeros(1, self.label_num+2, dtype=torch.long, device=device)
        val = torch.zeros(1, self.label_num+2, device=device)
        pre = emission_score_expand[0]

        for obs in emission_score_expand[1:]:
            pre_extend = pre.reshape(-1, 1).expand([self.label_num+2, self.label_num+2])
            obs_extend = obs.expand([self.label_num+2, self.label_num+2])
            score = obs_extend + pre_extend + self.transition_scores
            value, index = score.max(dim=0)
            ids = torch.cat([ids, index.unsqueeze(0)], dim=0)
            val = torch.cat([val, value.unsqueeze(0)], dim=0)
            pre = value

        index = torch.argmax(val[-1])
        best_path = [index]
        print('val[-1]:', val[-1])
        print('best_path:', best_path)

        for i in reversed(ids[1:]):
            index = i[index].item()
            best_path.append(index)
            print(i, 'best_path:', best_path)

        best_path = best_path[::-1][1:-1]

        return best_path


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, label_num):
        super(BiLSTM, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=256)
        self.blstm = nn.LSTM(
            input_size=256,
            hidden_size=512,
            bidirectional=True,
            num_layers=1
        )

        self.linear = nn.Linear(in_features=1024, out_features=label_num)

    def forward(self, inputs, length):
        outputs_embed = self.embed(inputs)
        outputs_packd = pack_padded_sequence(outputs_embed, length)
        outputs_blstm, (hn, cn) = self.blstm(outputs_packd)
        outputs_paded, outputs_lengths = pad_packed_sequence(outputs_blstm)
        outputs_paded = outputs_paded.transpose(0, 1)
        outputs_logits = self.linear(outputs_paded)
        outputs = []

        for outputs_logit, outputs_length in zip(outputs_logits, outputs_lengths):
            outputs.append(outputs_logit[:outputs_length])

        return outputs

    def predict(self, inputs):
        output_embed = self.embed(inputs)
        output_embed = output_embed.unsqueeze(1)
        # print('output_embed.shape1:', output_embed.shape)

        output_blstm, (hn, cn) = self.blstm(output_embed)
        output_blstm = output_blstm.squeeze(1)
        output_linear = self.linear(output_blstm)

        return output_linear


class NER(nn.Module):
    def __init__(self, vocab_size, label_num):
        super(NER, self).__init__()
        self.vocab_size = vocab_size
        self.label_num = label_num
        self.bilstm = BiLSTM(vocab_size=self.vocab_size, label_num=self.label_num).to(device)
        self.crf = CRF(label_num=self.label_num).to(device)

    def forward(self, inputs, labels, length):
        emission_scores = self.bilstm(inputs, length)
        batch_loss = self.crf(emission_scores, labels)

        return batch_loss

    def predict(self, inputs):
        # print('inputs.shape:', inputs.shape)
        emission_scores = self.bilstm.predict(inputs)
        logits = self.crf.predict(emission_scores)

        return logits

    def save_model(self, save_path):
        save_info = {
            'init': {'vocab_size': self.vocab_size, 'label_num': self.label_num},
            'state': self.state_dict()
        }
        torch.save(save_info, save_path)

if __name__ == '__main__':
    char_to_id = {"双": 0, "肺": 1, "见": 2, "多": 3, "发": 4, "斑": 5, "片": 6,
                  "状": 7, "稍": 8, "高": 9, "密": 10, "度": 11, "影": 12, "。": 13}
    tag_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}
    bilstm = BiLSTM(vocab_size=len(char_to_id),
               label_num=len(tag_to_id),)

    print(bilstm)
