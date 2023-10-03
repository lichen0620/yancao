import os
import sys
from itertools import chain

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

from models_tu import GAT_LSTM

from get_data_yc import setup_seed, test_data, scaler, node_feas, train_data, total_data


from args import gat_lstm_args_parser


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))


def get_r2(y, pred):
    return r2_score(y, pred)


def get_mae(y, pred):
    return mean_absolute_error(y, pred)


def get_mse(y, pred):
    return mean_squared_error(y, pred)


def get_rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))


@torch.no_grad()
def test(args, test_data, scaler):
    
    #graph_struct = []
    print('loading models...')
    model = GAT_LSTM(args).to(args.device)
    model_name = 'seq_len16hidden_size128output1model0'
    model.load_state_dict(torch.load('models/'+model_name+'.pkl')['model'])
    model.eval()
    print('predicting...')
    ys = [[] for i in range(args.input_size)]
    preds = [[] for i in range(args.input_size)]
    for graph in tqdm(test_data):
        graph = graph.to(args.device)
        _pred, targets, edge_inedx = model(graph, node_feas)     #8 4 1
        #graph_struct.append(edge_inedx)
        targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)  # 8 4 1
        #print(targets)                             #8 4 1
        #print(targets[:, 0, :])
        #z = list(chain.from_iterable(targets[:, 0, :]))
        #print(z)
        for i in range(args.input_size):
            target = targets[:, i, :]    # 8 1
            target = list(chain.from_iterable(target))
            ys[i].extend(target)
        for i in range(args.input_size):
            pred = _pred[:, i, :]
            pred = list(chain.from_iterable(pred.data.tolist()))
            preds[i].extend(pred)
    
    ys, preds = np.array(ys), np.array(preds)
    print(ys.shape)
    print(preds.shape)
    # with open('/home/laicx/GSL_GAT_LSTM_ing_two/txt/B18_pred.txt','w') as f:
    #     for i in preds[3]:
    #         f.write(str(i))
    #         f.write(',')
    #     f.close()
    # with open('/home/laicx/GSL_GAT_LSTM_ing_two/txt/B18_true.txt','w') as f:
    #     for i in ys[3]:
    #         f.write(str(i))
    #         f.write(',')
    #     f.close()      
    
    #ys = scaler.inverse_transform(ys)
    #preds = scaler.inverse_transform(preds)

    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        print('--------------------------------')
        print('第', str(ind), '个变量:')
        print('mse:', get_mse(y, pred))
        print('rmse:', get_rmse(y, pred))
        print('mae:', get_mae(y, pred))
        print('mape:', get_mape(y, pred))  

        print('--------------------------------')
        pred = pred[0:100]
        y = y[0:100]
        plot(y, pred, ind + 1,  model_name,label='*')

def plot(y, pred, ind, model_name,label):
    # plot
    fig = plt.figure()
    plt.plot(pred, color='blue', label='pred value')
    
    plt.plot(y, color='red', label='true value')
    plt.title('第' + str(ind) + '变量的预测示意图')
    plt.grid(True)
    #plt.legend(loc='upper center', ncol=6)
    #plt.show()
    fig.savefig('/home1/lichen/code/yancao_lc/outputs/'+model_name+'/第{}个变量({}).jpg'.format(ind, model_name))
    
args = gat_lstm_args_parser()
test(args=args, test_data=test_data, scaler=scaler)


