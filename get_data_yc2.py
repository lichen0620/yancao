import random
from pandas import Series

import scipy.io
import numpy as np
from datetime import datetime
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import os
import torch
from models import GAT_LSTM

from sklearn.preprocessing import MinMaxScaler
from torch import double, float64, nn
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric
from tqdm import tqdm
from args import gat_lstm_args_parser
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

args = gat_lstm_args_parser()

data_path = '/home1/lichen/code/yancao_lc/NASA/iot_telemetry_data.csv'

data = np.loadtxt(open(data_path,"rb"),delimiter=",",skiprows=1) 
data = data[0:5000,1:5]
#归一化
node_feas = torch.tensor(np.array(data)).float()
node_feas = list(map(list, zip(*node_feas)))
scaler = MinMaxScaler()
for i in range(len(node_feas)):
    node_feas[i] = scaler.fit_transform(np.array(node_feas[i]).reshape(-1, 1)).reshape(-1,)

data = torch.tensor(np.array(node_feas)).float()
data = list(map(list, zip(*node_feas)))

# edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 3, 3],
#   [2, 3, 0, 2, 3, 0, 0, 1]])    #全连接
edge_index = torch.tensor([[0, 0, 1, 2, 2, 3, 3, 3],
        [2, 3, 3, 0, 3, 0, 1, 2]])    #全连接
edge_index = torch.LongTensor(edge_index)                 


def process(dataset, batch_size, step_size, shuffle):
    seq = []
    graphs = []
    for i in tqdm(range(0, len(dataset) - args.seq_len - 1, step_size)):
        train_seq = []
        for j in range(i, i + args.seq_len):
            x = []
            for c in range(len(dataset[0])):  # 前8个时刻的所有变量
                x.append(dataset[j][c])
            train_seq.append(x)
        # 下1个时刻的所有变量
        train_labels = []
        for j in range(len(dataset[0])):
            train_label = []
            for k in range(i + args.seq_len, i + args.seq_len + 1):
                train_label.append(dataset[k][j])
            train_labels.append(train_label)
        # tensor
        train_seq = torch.FloatTensor(train_seq)
        train_labels = torch.FloatTensor(train_labels)
        #print(train_seq.shape, train_labels.shape)  # 8 4, 4 1

        # 此处可利用train_seq创建动态的邻接矩阵
        temp = Data(x=train_seq.T, y=train_labels, edge_index=edge_index)
        # print(temp)
        graphs.append(temp)
    train = graphs[:int(len(graphs) * 0.8)]
    val = graphs[int(len(graphs) * 0.6):int(len(graphs) * 0.8)]
    test = graphs[int(len(graphs) * 0.8):len(graphs)]
    train_data = torch_geometric.loader.DataLoader(train, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)
    val = torch_geometric.loader.DataLoader(val, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)
    test_data = torch_geometric.loader.DataLoader(test, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)
    total_data = torch_geometric.loader.DataLoader(graphs, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)
    return train_data, val, test_data, graphs, total_data
    # loader = torch_geometric.loader.DataLoader(graphs, batch_size=batch_size,
    #                                                 shuffle=shuffle, drop_last=False)
    # return loader
train_data, Val, test_data, graphs, total_data = process(data, batch_size=args.batch_size, step_size=1, shuffle=False)



