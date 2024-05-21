import copy, os, sys, yaml
sys.path.append(os.getcwd())
import pandas as pd
import torch.nn as nn
import torch as tr

from torch.utils.data import DataLoader
from experiment_pred_oracle.PredNet import FC_PredNet as OracleNet
from experiment_hc.PredNet import FC_PredNet as AttentionNet
from experiment_only.PredNet import Only_lstm_PredNet
from utils import dataset
from utils.tools import *

BASE_DIR = '/data/results/'
DATA_DIR = ['/data/hoyun_log/preprocess/dgx/storage/',
            '/data/betray_log/preprocessed_data/storage/',
            '/data/betray_log/preprocessed_data/storage2/',
            '/data/complex_log/preprocessed_data/storage/',
            'dummy'
            ]
DATA = ['test', 'betray', 'keep', 'complex','dummy']

EXPERIMENT = {
    #'oracle_simple' : {'config' : './configs/config.yaml',
    #            'network' : 'OracleNet',
    #            'saved_dir' : os.path.join(BASE_DIR, './oracle_simple/'),
    #            },
    #'onlylstm' : {'config' : './configs/config.yaml',
    #               'network' : 'Only_lstm_PredNet',
    #               'saved_dir' : './onlylstm/',
    #            },
    'attention' : {'config' : './configs/config.yaml',
                   'network' : 'AttentionNet',
                   'saved_dir' : os.path.join(BASE_DIR, './attention/'),
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
        with open(EXPERIMENT[model]['config']) as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            if model is not 'attention':
                configs.update({'charnet_type' : 'fc', 'no_char' : True})
            else:
                configs.update({'charnet_type' : 'fc', 'no_char' : False})
        saved_dir= EXPERIMENT[model]['saved_dir']
        net = eval(EXPERIMENT[model]['network'])(device=device, configs=configs, norm_adjacency=norm_adjacency)
        net.to('cuda')
        
        for saved_model in os.listdir(EXPERIMENT[model]['saved_dir']):
            net.load_state_dict(tr.load(saved_dir + saved_model))
            for name, data in zip(DATA, DATA_DIR):
                tom_epoch_order_acc = 0
                tom_epoch_dst_acc = 0
                tom_epoch_response_acc = 0
                if data=='dummy':
                    tom_dataset = dataset.make_dataset(DATA_DIR[0], configs['exp'], 0, 1)
                else:
                    num = 108 if name != 'test' else 0
                    tom_dataset = dataset.make_dataset(data, configs['exp'], num, num + 1)
                dataloader = DataLoader(tom_dataset, batch_size=128, shuffle=False)
                net.eval()
                for i, batch in enumerate(dataloader):
                    if model == 'attention':
                        p_board, p_order, p_message, c_board, c_src, c_message, t_order, t_dst, t_response, \
                        me_weights, other_weights, me_ind, other_ind, c_send, c_order = batch
                        p_board = p_board.float().cuda()
                        p_order = p_order.float().cuda()
                        p_message = p_message.float().cuda()
                        if data=='dummy':
                           p_message = tr.zeros(p_message.shape).float().cuda()
                    else:
                        c_board, c_src, c_message, t_order, t_dst, t_response, \
                            me_weights, other_weights, me_ind, other_ind, c_send, c_order, c_other_internal = batch
                    c_board = c_board.float().cuda()
                    c_src = c_src.float().cuda()
                    c_message = c_message.float().cuda()
                    if data=='dummy':
                        c_message = tr.zeros(c_message.shape).float().cuda()
                    c_send = c_send.float().cuda()
                    c_order = c_order.float().cuda()
                    t_order = tr.flatten(t_order.long().cuda())
                    t_dst = tr.flatten(t_dst.long().cuda())
                    t_response = tr.flatten(t_response.long().cuda())
                    if model == 'oracle_simple':
                        pred_order, pred_dst, pred_response, e_char = net(c_board, c_src, c_message, c_send, c_order, me_weights, other_weights, me_ind, other_ind, c_other_internal)
                    elif model == 'onlylstm':
                        pred_order, pred_dst, pred_response, e_char = net(c_board, c_src, c_message, c_send, c_order)
                    elif model == 'attention':
                        pred_order, pred_dst, pred_response, e_char, weights = net(p_board, p_order, p_message, c_board, c_src, c_message, c_send,  c_order)
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

                result[f'{model}_{name}_order_acc'].append(tom_epoch_order_acc/len(tom_dataset))
                result[f'{model}_{name}_dst_acc'].append(tom_epoch_dst_acc/len(tom_dataset))
                result[f'{model}_{name}_response_acc'].append(tom_epoch_response_acc/len(tom_dataset))
        
    result_df = pd.DataFrame(result).T.reset_index()
    result_df.columns = ['experiment', 'run0', 'run1']
    result_df['mean'] = result_df.mean(axis=1)
    result_df['std'] = result_df.std(axis=1)
    
    #print(result_df.round(4).to_markdown())
    result_df['model'], result_df['data'], result_df['target'], _ = result_df['experiment'].str.split('_').str
    result_df['view'] = result_df['mean'].round(4).astype(str) + '+-' +(1.96 * result_df['std']/ 5**(1/2)).round(4).astype(str)
    result_df.pivot(index = ['data', 'target'], values = 'view', columns = 'model').to_csv("clarified_results.csv")
    result_df.to_csv("result.csv")

if __name__ == '__main__':
    run_test()