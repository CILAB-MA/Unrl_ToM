import copy, os, sys, yaml
sys.path.append(os.getcwd())
import pandas as pd
import torch.nn as nn
import torch as tr

import torch.nn.functional as F
from torch.utils.data import DataLoader
from experiments import dataset
from experiments.tomnet.PredNet import FC_PredNet as ToMNet
from experiments.oracle.PredNet import FC_PredNet as OracleNet
from experiments.attention.PredNet import FC_PredNet as AttentionNet
from experiments.only_lstm.PredNet import Only_lstm_PredNet
from experiments.tools import *

BASE_DIR = '/workspace/uwonsang/data/test/'
CONFIG_DIR = './experiments/config.yaml'
DATA = ['test']#, 'betray', 'complex','dummy']

EXPERIMENT = {
    # 'tomnet' : {
    #             'network' : 'ToMNet',
    #             'saved_dir' : './tom/',
    #             },
    # 'oracle' : {
    #             'network' : 'OracleNet',
    #             'saved_dir' : './oracle/',
    #             },
    'only_lstm' : {
                   'network' : 'Only_lstm_PredNet',
                   'saved_dir' : './onlylstm/',
                },
    'attention' : {
                   'network' : 'AttentionNet',
                   'saved_dir' : './attention/',
                }
}


def run_test():
    norm_adjacency = preprocess_adjacency(get_adjacency_matrix('standard'))
    device = 'cuda' if tr.cuda.is_available() else 'cpu'
    
    result = {}
    for m in EXPERIMENT.keys():
        for d in DATA:
            result.update({f"{m}_{d}_order_acc" : [],
                           f"{m}_{d}_dst_acc" : [],
                           f"{m}_{d}_response_acc" : []})

    for model in EXPERIMENT.keys():
        with open(CONFIG_DIR, 'r') as f:
            all_configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = dict(all_configs["basic"], **all_configs["{}".format(model)])
        if model == 'attention':
            configs.update({'no_char' : False})
        else:
            configs.update({'no_char' : True})
        #args
        configs.update({'exp': model, 'charnet_type' : 'fc'})

        saved_dir= EXPERIMENT[model]['saved_dir']
        net = eval(EXPERIMENT[model]['network'])(device=device, configs=configs, norm_adjacency=norm_adjacency)
        net.to('cuda')
        
        for saved_model in os.listdir(EXPERIMENT[model]['saved_dir']):
            net.load_state_dict(tr.load(saved_dir + saved_model))
            for data in DATA:
                tom_epoch_order_acc = 0
                tom_epoch_dst_acc = 0
                tom_epoch_response_acc = 0
                data_folder = BASE_DIR+f'test' if data=='dummy' else BASE_DIR+f'{data}' 
                datalen = len(os.listdir(data_folder))
                tom_dataset = dataset.make_dataset(data_folder, configs['exp'], configs['input_infos'],  0, datalen)
                dataloader = DataLoader(tom_dataset, batch_size=128, shuffle=False)
                net.eval()
                for i, batch in enumerate(dataloader):
                    if model == 'attention':
                        p_board, p_order, p_message, c_board, c_src, c_message, t_order, t_dst, t_response, \
                            me_weights, other_weights, me_ind, other_ind, c_send, c_order = batch
                    
                        p_board = p_board.float().cuda()
                        p_order = p_order.float().cuda()
                        p_message = p_message.float().cuda()
                        me_ind = me_ind.float().cuda()
                        other_ind = other_ind.float().cuda()
                        if data=='dummy':
                           p_message = tr.zeros(p_message.shape).float().cuda()
                    else:
                        c_board, c_src, c_message, t_order, t_dst, t_response, \
                            me_weights, other_weights, me_ind, other_ind, c_send, c_order = batch#, c_other_internal = batch
                        me_ind = F.one_hot(me_ind[:, -1].to(tr.int64), num_classes=7)
                        me_ind = me_ind.reshape(-1, 7).unsqueeze(1).float().to(device)
                        other_ind = F.one_hot(other_ind[:, -1].to(tr.int64), num_classes=7)
                        other_ind = other_ind.reshape(-1, 7).unsqueeze(1).to(device)
                    c_board = c_board.float().to(device)
                    c_src = c_src.float().to(device)
                    c_message = c_message.float().to(device)
                    if data=='dummy':
                        c_message = tr.zeros(c_message.shape).float().cuda()

                    c_send = c_send.float().to(device)
                    c_order = c_order.float().to(device)
                    t_order = tr.flatten(t_order.long().to(device))
                    t_dst = tr.flatten(t_dst.long().to(device))
                    t_response = tr.flatten(t_response.long().to(device))
                    
                    if model == 'oracle':
                        pred_order, pred_dst, pred_response, e_char = net(c_board, c_src, c_message, c_send, c_order, me_weights, other_weights, me_ind, other_ind, c_other_internal)
                    elif model == 'only_lstm':
                        pred_order, pred_dst, pred_response, e_char = net(c_board, c_src, c_message, c_send, c_order, me_ind, other_ind)
                    elif model == 'attention':
                        pred_order, pred_dst, pred_response, e_char, weights = net(p_board, p_order, p_message, c_board, c_src, c_message, c_send,  c_order, other_ind, me_ind)
                    else:
                        raise Exception('invalid model')

                    tom_order_inds = tr.argmax(pred_order, dim=-1)
                    tom_dst_inds = tr.argmax(pred_dst, dim=-1)
                    tom_response_inds = tr.argmax(pred_response, dim=-1)

                    tom_order_acc = tr.sum(tom_order_inds == t_order).item()
                    tom_dst_acc = tr.sum(tom_dst_inds == t_dst).item()
                    tom_response_acc = tr.sum(tom_response_inds == t_response).item()

                    tom_epoch_order_acc += tom_order_acc
                    tom_epoch_dst_acc += tom_dst_acc
                    tom_epoch_response_acc += tom_response_acc

                result[f'{model}_{data}_order_acc'].append(tom_epoch_order_acc/len(tom_dataset))
                result[f'{model}_{data}_dst_acc'].append(tom_epoch_dst_acc/len(tom_dataset))
                result[f'{model}_{data}_response_acc'].append(tom_epoch_response_acc/len(tom_dataset))
        
    result_df = pd.DataFrame(result).T.reset_index()
    # result_df.columns = ['experiment', 'run0', 'run1','run2','run3', 'run4']
    # result_df['mean'] = result_df.mean(axis=1)
    # result_df['std'] = result_df.std(axis=1)
    
    # #print(result_df.round(4).to_markdown())
    # result_df['model'], result_df['data'], result_df['target'], _ = result_df['experiment'].str.split('_').str
    # result_df['view'] = result_df['mean'].round(4).astype(str) + '+-' +(1.96 * result_df['std']/ 5**(1/2)).round(4).astype(str)
    # result_df.pivot(index = ['data', 'target'], values = 'view', columns = 'model').to_csv("clarified_results.csv")
    # result_df.to_csv("result.csv")

if __name__ == '__main__':
    run_test()