from collections import Counter
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import os
import time
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from models import GAT_LSTM
from args import gat_lstm_args_parser
#from get_data_GSL import setup_seed, train_data, Val, node_feas
from get_data_yc import setup_seed, train_data, Val, node_feas
#from get_data import setup_seed, train_data, Val, node_feas


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
setup_seed(0)

def get_val_loss(args, model, Val, node_feas):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for graph in Val:
        graph = graph.to(args.device)
        preds, labels = model(graph, node_feas)
        #total_loss = 0
        #for k in range(args.input_size):
            #total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])
        #total_loss /= preds.shape[0]
        total_loss = loss_function(preds, labels)
        val_loss.append(total_loss.item())

    return np.mean(val_loss)


def train(args, train_data, val):
    
    model = GAT_LSTM(args).to(args.device)
    model.train()
    loss_function_1 = nn.MSELoss().to(args.device)
    loss_function_2 = nn.L1Loss().to(args.device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 2
    best_model = None
    min_val_loss = 5
    loss = []
    graph_struct = []
    for epoch in tqdm(range(args.epochs),ncols=10):
        time.sleep(0.001)
        train_loss = []
        epoch_loss = 0
        for graph in train_data:
            graph = graph.to(args.device)
            preds, labels, edge_index = model(graph, node_feas)
            graph_struct.append(edge_index)
            #print(preds)
            #print(preds.shape)
            #print(labels)
            #print(labels.shape)
            total_loss = loss_function_1(preds, labels)+loss_function_2(preds, labels)
            epoch_loss = epoch_loss + total_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())
        
            
        loss.append(epoch_loss / len(train_data))
        scheduler.step()
        # validation
        # val_loss = get_val_loss(args, model, val, node_feas)
        # if epoch + 1 >= min_epochs and val_loss < min_val_loss:
        #     min_val_loss = val_loss
        #     best_model = copy.deepcopy(model)
        #     state = {'model': best_model.state_dict()}
        #     torch.save(state, 'models/' + 'model_version_{}'.format(args.model_version) + '.pkl')

        #print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        print('epoch {:03d} train_loss {:.8f} '.format(epoch, np.mean(train_loss)))
        #model.train()
        
    #######################################################################################
    # 计算出现最多的图结构
    print(graph_struct)
    print(len(graph_struct))
    print('最后一个图结构为:{}'.format(graph_struct[-1]))
    
    collection_graph = Counter(graph_struct)
    most_counterNum = collection_graph.most_common(3)
    # print(collection_graph)
    # print('出现最多的三种图结构为{}'.format(most_counterNum))
    
    # #######################################################################################
    # count = []
    # print('search for graph struct that happens most')
    # for i in tqdm(range(len(graph_struct))):
    #     num = 0
    #     for j in range(len(graph_struct)):
    #         if torch.equal(graph_struct[i], graph_struct[j]):
    #             num += 1
    #     count.append(num)
        
    # max_value = max(count)
    
    # print(count)
    
    # idx = [i for i,x in enumerate(count) if x==max_value]

    # print(max_value)
    # print(idx)
    # for i in idx:
    #     print(graph_struct[i])
   ############################################################################################
    loss = torch.tensor(loss, device = 'cpu')
    plt.plot(loss, color='red', label='total_loss')
    plt.title('全过程loss')
    plt.grid(True)
    #plt.legend(loc='upper center', ncol=6)
    plt.show()
    plt.savefig('/home1/lichen/code/yancao_lc/outputs/all_loss(seq_len={})_hidden_size{}_model{}.jpg'.format(args.seq_len,args.hidden_size,args.model_version))
        #if epoch % 10 == 0:
            #print('epoch={} | loss={} '.format(epoch, train_loss))
        
    state = {'model': model.state_dict()}
    torch.save(state, 'models/' + 'seq_len{}_hidden_size{}_model{}'.format(args.seq_len,args.hidden_size,args.model_version) + '.pkl')
    

args = gat_lstm_args_parser()
if __name__ == '__main__':
    train(args, train_data=train_data, val=Val)


