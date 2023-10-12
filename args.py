# -*- coding:utf-8 -*-
import argparse
import torch





def gat_lstm_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--input_size', type=int, default=4, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=16, help='seq len')#################################
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')  ########1试试
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--model_version', type=int, default=0, help='model_version')
    parser.add_argument('--lamda', type=float, default=1, help='ratio of lossL1')

    args = parser.parse_args()

    return args


