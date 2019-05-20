import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layer import EncoderLayer, KeyEncoderLayer, AttBasedLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.drop_out = nn.Dropout(0.5)
        self.bidirectional = 2 if bidirectional else 1
        self._gru = nn.GRU(input_size=input_dim, hidden_size=n_hidden, num_layers=n_layer,
                            batch_first = True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, text, input_lens=None):
        """[batch_size, max_len, input_dim] Tensor"""
        if self.training:
            text = self.drop_out(text)
        if input_lens is not None:
            sorted_len, indices = torch.sort(input_lens, 0, descending=True)
            text = text[indices]
            _, reversed_indices = torch.sort(indices, 0)
            packed_text = pack_padded_sequence(text, sorted_len, batch_first=True)
            pack_output, h = self._gru(packed_text)
            gru_output, _ = pad_packed_sequence(pack_output)
            h = h[-1]
            h = h[reversed_indices]
        else:
            gru_output, h = self._gru(text)

        return h

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, gru_layer, bidirectional,
                  d_inner, n_head, d_k, d_v, n_layers, dropout=0.1):
        super().__init__()
        self._d_model = hidden_dim
        self.layer_stack = nn.ModuleList([
            EncoderLayer(self._d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
    def forward(self, src_seq, seq_len):

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output)

        return enc_output

class KeyEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, gru_layer, bidirectional,
                  d_inner, n_head, d_k, d_v, n_layers, dropout=0.1, mask=True, init_embedding=None, hop=3):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        if init_embedding is not None:
            self._embedding.weight = nn.Parameter(torch.FloatTensor(init_embedding).cuda())
        self._gru = GRUEncoder(embed_dim, hidden_dim, gru_layer, 0.5, bidirectional)
        self._mask = mask
        if mask:
            self._maskEncoder = Encoder(vocab_size, embed_dim, hidden_dim, gru_layer, bidirectional,
                  d_inner, n_head, d_k, d_v, n_layers)
            self._maskW1 = nn.Linear(hidden_dim, d_k)
            self._maskW2 = nn.Linear(hidden_dim, d_k)
            self._masktemperature = np.power(d_k, 0.5)
        self._d_model = hidden_dim
        self._hop = hop
        self.layer_stack = nn.ModuleList([
            KeyEncoderLayer(self._d_model, d_inner, n_head, d_k, d_v, dropout=0.2)
            for _ in range(n_layers)])
        self.ti_att = nn.ModuleList([
            AttBasedLayer(hidden_dim, d_inner, n_head, d_k, d_v, dropout=0.2)
            for _ in range(self._hop)])
        self.ti_ln = nn.LayerNorm(hidden_dim)
        self.my_att = nn.ModuleList([
            AttBasedLayer(hidden_dim, d_inner, n_head, d_k, d_v, dropout=0.2)
            for _ in range(self._hop)])
        self.my_ln = nn.LayerNorm(hidden_dim)
        self.fri_att = nn.ModuleList([
            AttBasedLayer(hidden_dim, d_inner, n_head, d_k, d_v, dropout=0.2)
            for _ in range(self._hop)])
        self.fri_ln = nn.LayerNorm(hidden_dim)
        self.mf_ln = nn.LayerNorm(hidden_dim)
        self.re_drop = nn.Dropout(0.1)


        self._predict = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, rt_seq, rt_len, timeline, timeline_seq_len, my_hist,\
            my_hist_seq_len, friend_hist, friend_hist_seq_len, activation):


        rt_seq = rt_seq.cuda()
        rt_len = rt_len.cuda()
        timeline = timeline.cuda()
        timeline_seq_len = timeline_seq_len.cuda()
        my_hist = my_hist.cuda()
        my_hist_seq_len = my_hist_seq_len.cuda()
        friend_hist = friend_hist.cuda()
        friend_hist_seq_len = friend_hist_seq_len.cuda()


        rt_seq = self._embedding(rt_seq)
        rt_seq = self._gru(rt_seq, rt_len)
        timeline = self._embedding(timeline)
        my_hist = self._embedding(my_hist)
        friend_hist = self._embedding(friend_hist)

        ti_output = []
        for i in range(timeline.size(0)):
            ti_output.append(self._gru(timeline[i], timeline_seq_len[i]))
        ti_output = torch.stack(ti_output, dim=0)

        my_output = []
        for i in range(my_hist.size(0)):
            my_output.append(self._gru(my_hist[i], my_hist_seq_len[i]))
        my_output = torch.stack(my_output, dim=0)

        fri_output = []
        for i in range(friend_hist.size(0)):
            fri_output.append(self._gru(friend_hist[i], friend_hist_seq_len[i]))
        fri_output = torch.stack(fri_output, dim=0)

        if self._mask:
            mask_output = self._maskEncoder(ti_output, timeline_seq_len)
            mask_W1 = self._maskW1(rt_seq)
            mask_W2 = self._maskW2(mask_output)

            if activation == 'relu6':
                mask_output = F.relu6(torch.bmm(mask_W2, mask_W1.unsqueeze(2)) / self._masktemperature)
            elif activation == 'sigmoid':
                mask_output = F.sigmoid(torch.bmm(mask_W2, mask_W1.unsqueeze(2)) / self._masktemperature)
            else:
                mask_output = F.threshold(torch.bmm(mask_W2, mask_W1.unsqueeze(2)) / self._masktemperature, threshold=0.5, value=0)

        for enc_layer in self.layer_stack:
            if self._mask:
                ti_output = mask_output*ti_output
            ti_output, enc_slf_attn = enc_layer(ti_output, ti_output)
        rt_ti = rt_seq
        for att_layer in self.ti_att:
            t_output, _ = att_layer(rt_ti.unsqueeze(1), ti_output, ti_output)
            rt_ti = self.ti_ln(self.re_drop(rt_ti + t_output))

        rt_mf = rt_seq
        for i in range(self._hop):
            matt_layer = self.my_att[i]
            m_output, _ = matt_layer(rt_mf.unsqueeze(1), my_output, my_output)
            fatt_layer = self.fri_att[i]
            f_output, _ = fatt_layer(rt_mf.unsqueeze(1), fri_output, fri_output)
            rt_mf = self.mf_ln(self.re_drop(rt_mf + m_output + f_output))

        enc_output = self._predict(torch.cat([rt_ti, rt_mf], dim=1))
        return enc_output
