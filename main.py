from model.mbt import Encoder, KeyEncoder
from data_utils.batch_helper import get_batch
import pickle
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from utils.result_calculator import *
import argparse
from data_utils.load_data import load_embedding
from torch.nn.utils import clip_grad_norm_
torch.cuda.manual_seed_all(999)
torch.manual_seed(999)

parser = argparse.ArgumentParser()
parser.add_argument('--beta', type=float, default=0.7, help='beta factor (default: 0.7)')
parser.add_argument('--drop_out', type=float, default=0.5, help='drop_out (default: 0.5)')
parser.add_argument('--batch_size', type=int, default=150, help='batch_size (default: 150)')
parser.add_argument('--nums', type=int, default=30, help='length of tweets (default: 30)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate (default: 0.003)')
parser.add_argument('--embedding', type=int, default=300, help='embedding_dim (default: 300)')
parser.add_argument('--hidden_dim', type=int, default=100, help='hidden_dim (default: 100)')
parser.add_argument('--gru_layer', type=int, default=3, help='gru_layer (default: 3)')
parser.add_argument('--bidirectional', type=int, default=1, help='bidirectional (default: True)')
parser.add_argument('--mask', type=int, default=1, help='mask (default: True)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay (default: 0.0005)')
parser.add_argument('--clip', type=float, default=0.7, help='clip (default: 0.7)')
parser.add_argument('--point', type=int, default=0, help='save point (default: 0)')
parser.add_argument('--activation', type=str, default='sigmoid', help='the activation function be chosen to ')
parser.add_argument('--hop', type=int, default=3, help='hop (default: 3)')
args = parser.parse_args()

args.bidirectional = True if args.bidirectional == 1 else False
args.mask = True if args.mask == 1 else False

batch_size = args.batch_size
log_interval = 20
epoches = 24
drop_out = args.drop_out
embedding_dim = args.embedding
beta = args.beta
weight_decay = args.weight_decay
learningrate = args.lr
point = args.point
clip = args.clip

train_set = pickle.load(open("../processed_data/train.pkl", "rb"))
valid_set = pickle.load(open("../processed_data/valid.pkl", "rb"))
test_set = pickle.load(open("../processed_data/test.pkl", "rb"))
vocab = pickle.load(open("../processed_data/vocab.pkl", "rb"))



init_embedding = load_embedding(vocab, args.embedding)

net = KeyEncoder(len(vocab), args.embedding, args.hidden_dim, args.gru_layer, args.bidirectional,
              d_inner=2048, n_head=8, d_k=64, d_v=64, n_layers=6, dropout=args.drop_out, mask=args.mask,
                 init_embedding=init_embedding, hop=args.hop)
net.cuda()

load_path = 'parameters/parameter.h5'

best_p, best_r, best_f1 = 0.0, 0.0, 0.0
net.load_state_dict(torch.load(load_path))

net.eval()
with torch.no_grad():
    valid_predict = []
    valid_target = []
    test_predict = []
    test_target = []
    for d in get_batch(valid_set, batch_size, args.nums, shuffle=False):
        rt, rt_mask, rt_seq_len, timeline, time_mask, timeline_seq_len, my_hist, my_hist_mask, \
        my_hist_seq_len, friend_hist, friend_hist_mask, friend_hist_seq_len, target = d
        target = torch.Tensor(target.tolist()).long().cuda()
        predict = net(rt, rt_seq_len, timeline, timeline_seq_len, my_hist,\
        my_hist_seq_len, friend_hist, friend_hist_seq_len, args.activation)
        valid_predict.extend(predict.tolist())
        valid_target.extend(target.tolist())
    valid_precision = precision_score(np.asarray(valid_target), np.argmax(np.asarray(valid_predict), 1))
    valid_recall = recall_score(np.asarray(valid_target), np.argmax(np.asarray(valid_predict), 1))
    valid_f1 = 0.0
    if valid_precision + valid_recall == 0:
        valid_f1 = 0.0
    else:
        valid_f1 = 2*valid_precision*valid_recall / (valid_precision + valid_recall)
    print('valid_precision:', valid_precision, 'valid_recall:', valid_recall, 'valid_f1:', valid_f1)
    for d in get_batch(test_set, batch_size, args.nums, shuffle=False):
        rt, rt_mask, rt_seq_len, timeline, time_mask, timeline_seq_len, my_hist, my_hist_mask, \
        my_hist_seq_len, friend_hist, friend_hist_mask, friend_hist_seq_len, target = d
        target = torch.Tensor(target.tolist()).long().cuda()
        predict = net(rt, rt_seq_len, timeline, timeline_seq_len, my_hist,\
        my_hist_seq_len, friend_hist, friend_hist_seq_len, args.activation)
        test_predict.extend(predict.tolist())
        test_target.extend(target.tolist())
    test_precision = precision_score(np.asarray(test_target), np.argmax(np.asarray(test_predict), 1))
    test_recall = recall_score(np.asarray(test_target), np.argmax(np.asarray(test_predict), 1))
    test_f1 = 2*test_precision*test_recall / (test_precision + test_recall)
    print('test_precision:', test_precision, 'test_recall:', test_recall, 'test_f1:', test_f1)