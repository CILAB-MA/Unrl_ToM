import copy, os, sys, yaml
sys.path.append(os.getcwd())
import pandas as pd
import torch.nn as nn
import torch as tr

from torch.utils.data import DataLoader
from experiments.attention.PredNet import FC_PredNet as ATTENTION
from experiments.tomnet.PredNet import FC_PredNet as TOMNET
from experiments.oracle.PredNet import FC_PredNet as ORACLE
from experiments.only_lstm.PredNet import Only_lstm_PredNet as ONLYLSTM
from experiments import dataset
from experiments.tools import *



BASE_DIR = '/data/'

DATA = ['test']

EXPERIMENT1 = {
    'attention' : {'config' : './experiments/config.yaml',
                   'network' : ATTENTION,
                   'saved_dir' : './attention/',
                },
    'tomnet': {'config': './experiments/config.yaml',
                  'network': TOMNET,
                  'saved_dir': './tomnet/',
                  }
}

EXPERIMENT2 = {
    'oracle_simple' : {'config' : './configs/config.yaml',
                'network' : ORACLE,
                'saved_dir' : './oracle_simple/',
                },
    'onlylstm' : {'config' : './configs/config.yaml',
                   'network' : ONLYLSTM,
                   'saved_dir' : './onlylstm/',
                },
    'attention' : {'config' : './configs/config.yaml',
                   'network' : ATTENTION,
                   'saved_dir' : './attention/',
                },
    'tomnet': {'config': './configs/config.yaml',
                  'network': TOMNET,
                  'saved_dir': './tomnet/',
                  }
}
def run_num_past_test():
    past_configs = [1, 2, 3]
    result = {}
    norm_adjacency = preprocess_adjacency(get_adjacency_matrix('standard'))
    device = 'cuda' if tr.cuda.is_available() else 'cpu'
    for model in EXPERIMENT1.keys():
        with open(EXPERIMENT1[model]['config']) as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            configs = dict(configs["basic"], **configs["{}".format(model)])
            configs.update({'charnet_type': 'fc', 'no_char': False ,'exp': model})
        saved_dir = EXPERIMENT1[model]['saved_dir']
        net = EXPERIMENT1[model]['network'](device=device, configs=configs, norm_adjacency=norm_adjacency)
        net.to('cuda')
        for saved_model in os.listdir(EXPERIMENT1[model]['saved_dir']):
            net.load_state_dict(tr.load(saved_dir + saved_model))
            print('MODEL UPLOADED!!', saved_model)
            tom_epoch_order_acc = 0
            tom_epoch_dst_acc = 0
            tom_epoch_response_acc = 0
            print(configs, model)
            tom_dataset = dataset.make_dataset(BASE_DIR + f'test_data/', configs['exp'], configs['input_infos'], 19, 20)
            dataloader = DataLoader(tom_dataset, batch_size=128, shuffle=False)
            net.eval()

            for i, batch in enumerate(dataloader):

                p_board, p_order, p_message, c_board, c_src, c_message, t_order, t_dst, t_response, \
                me_weights, other_weights, me_ind, other_ind, c_send, c_order = batch
                p_board = p_board.float().cuda()
                p_order = p_order.float().cuda()
                p_message = p_message.float().cuda()
                c_board = c_board.float().cuda()
                c_src = c_src.float().cuda()
                c_message = c_message.float().cuda()
                c_send = c_send.float().cuda()
                c_order = c_order.float().cuda()
                t_order = tr.flatten(t_order.long().cuda())
                t_dst = tr.flatten(t_dst.long().cuda())
                t_response = tr.flatten(t_response.long().cuda())

                for p in past_configs:
                    p_board_tmp = p_board[:, :p]
                    p_order_tmp = p_order[:, :p]
                    p_message_tmp = p_message[:, :p]
                    pred_order, pred_dst, pred_response, e_char, weights = net(p_board_tmp, p_order_tmp, p_message_tmp,
                                                                               c_board, c_src, c_message, c_send, c_order)

                    tom_order_inds = tr.argmax(pred_order, dim=-1)
                    tom_dst_inds = tr.argmax(pred_dst, dim=-1)
                    tom_response_inds = tr.argmax(pred_response, dim=-1)

                    tom_order_acc = tr.sum(tom_order_inds == t_order).item()
                    tom_dst_acc = tr.sum(tom_dst_inds == t_dst).item()
                    tom_response_acc = tr.sum(tom_response_inds == t_response).item()

                    tom_epoch_order_acc += tom_order_acc
                    tom_epoch_dst_acc += tom_dst_acc
                    tom_epoch_response_acc += tom_response_acc

                    result[f'{model}_past_{p}_order_acc'].append(tom_epoch_order_acc / len(tom_dataset))
                    result[f'{model}_past_{p}_dst_acc'].append(tom_epoch_dst_acc / len(tom_dataset))
                    result[f'{model}_past_{p}_response_acc'].append(tom_epoch_response_acc / len(tom_dataset))

        result_df = pd.DataFrame(result).T.reset_index()
        result_df.columns = ['experiment', 'run0', 'run1', 'run2', 'run3']
        result_df['mean'] = result_df.mean(axis=1)
        result_df['std'] = result_df.std(axis=1)

        # print(result_df.round(4).to_markdown())
        result_df['model'], result_df['data'], result_df['target'], _ = result_df['experiment'].str.split('_').str
        result_df['view'] = result_df['mean'].round(4).astype(str) + '+-' + (
                    1.96 * result_df['std'] / 5 ** (1 / 2)).round(4).astype(str)
        result_df.pivot(index=['data', 'target'], values='view', columns='model').to_csv("clarified_results.csv")
        result_df.to_csv("result_past_traj.csv")



def run_curr_step_test():

    norm_adjacency = preprocess_adjacency(get_adjacency_matrix('standard'))
    device = 'cuda' if tr.cuda.is_available() else 'cpu'

    result = {}
    for m in EXPERIMENT2.keys():
        for d in DATA:
            result.update({f"{m}_{d}_order_acc" : [],
                           f"{m}_{d}_dst_acc" : [],
                           f"{m}_{d}_response_acc" : []})

    curr_configs = [1, 10, 20, 30, 40]

    for model in EXPERIMENT2.keys():
        with open(EXPERIMENT2[model]['config']) as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            if model is not 'attention':
                configs.update({'charnet_type' : 'fc', 'no_char' : True})
            else:
                configs.update({'charnet_type' : 'fc', 'no_char' : False})
        saved_dir= EXPERIMENT2[model]['saved_dir']
        net = MODULES[EXPERIMENT2[model]['network']](device=device, configs=configs, norm_adjacency=norm_adjacency)
        net.to('cuda')
        for saved_model in os.listdir(EXPERIMENT2[model]['saved_dir']):
            net.load_state_dict(tr.load(saved_dir + saved_model))
            tom_epoch_order_acc = 0
            tom_epoch_dst_acc = 0
            tom_epoch_response_acc = 0
            tom_dataset = dataset.make_dataset(BASE_DIR+f'test_data/', configs['exp'], 0, 2)
            dataloader = DataLoader(tom_dataset, batch_size=128, shuffle=False)
            net.eval()
            for i, batch in enumerate(dataloader):
                if model == 'attention':
                    p_board, p_order, p_message, c_board, c_src, c_message, t_order, t_dst, t_response, \
                    me_weights, other_weights, me_ind, other_ind, c_send, c_order = batch
                    p_board = p_board.float().cuda()
                    p_order = p_order.float().cuda()
                    p_message = p_message.float().cuda()
                else:
                    c_board, c_src, c_message, t_order, t_dst, t_response, \
                        me_weights, other_weights, me_ind, other_ind, c_send, c_order, c_other_internal = batch

                c_board = c_board.float().cuda()
                c_src = c_src.float().cuda()
                c_message = c_message.float().cuda()
                c_order = c_order.float().cuda()
                c_send = c_send.float().cuda()
                t_order = tr.flatten(t_order.long().cuda())
                t_dst = tr.flatten(t_dst.long().cuda())
                t_response = tr.flatten(t_response.long().cuda())

                for c in curr_configs:
                    c_board_tmp = c_board[:, -c:]
                    c_order_tmp = c_order[:, -c:]
                    c_message_tmp = c_message[:, -c:]
                    # PLZ CHECK FOR STEP IS RIGHT!

                    if model == 'oracle_simple':
                        pred_order, pred_dst, pred_response, e_char = net(c_board_tmp, c_src, c_message_tmp, c_send, c_order_tmp, me_weights, other_weights, me_ind, other_ind, c_other_internal)
                    elif model == 'onlylstm':
                        pred_order, pred_dst, pred_response, e_char = net(c_board_tmp, c_src, c_message_tmp, c_send, c_order_tmp)
                    elif model in ['attention', 'tomnet']:
                        pred_order, pred_dst, pred_response, e_char, weights = net(p_board, p_order, p_message, c_board_tmp, c_src, c_message_tmp, c_send,  c_order_tmp)
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

                    result[f'{model}_curr_{c}_order_acc'].append(tom_epoch_order_acc/len(tom_dataset))
                    result[f'{model}_curr_{c}_dst_acc'].append(tom_epoch_dst_acc/len(tom_dataset))
                    result[f'{model}_curr_{c}_response_acc'].append(tom_epoch_response_acc/len(tom_dataset))

                    '''
                    dummy test 
                    if c == 40:
                        c_message = tr.zeros(c_message.shape).float().cuda()
                        if model in ['attention', 'tomnet']:
                            p_message = tr.zeros(p_message.shape).float().cuda()
                        if model == 'oracle_simple':
                            pred_order, pred_dst, pred_response, e_char = net(c_board_tmp, c_src, c_message_tmp, c_send,
                                                                              c_order_tmp, me_weights, other_weights,
                                                                              me_ind, other_ind, c_other_internal)
                        elif model == 'onlylstm':
                            pred_order, pred_dst, pred_response, e_char = net(c_board_tmp, c_src, c_message_tmp, c_send,
                                                                              c_order_tmp)
                        elif model in ['attention', 'tomnet']:
                            pred_order, pred_dst, pred_response, e_char, weights = net(p_board, p_order, p_message,
                                                                                       c_board_tmp, c_src,
                                                                                       c_message_tmp, c_send,
                                                                                       c_order_tmp)
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

                        result[f'{model}_curr_{c}_order_acc'].append(tom_epoch_order_acc / len(tom_dataset))
                        result[f'{model}_curr_{c}_dst_acc'].append(tom_epoch_dst_acc / len(tom_dataset))
                        result[f'{model}_curr_{c}_response_acc'].append(tom_epoch_response_acc / len(tom_dataset))
                    '''

    result_df = pd.DataFrame(result).T.reset_index()
    result_df.columns = ['experiment', 'run0', 'run1','run2','run3', 'run4']
    result_df['mean'] = result_df.mean(axis=1)
    result_df['std'] = result_df.std(axis=1)

    #print(result_df.round(4).to_markdown())
    result_df['model'], result_df['data'], result_df['target'], _ = result_df['experiment'].str.split('_').str
    result_df['view'] = result_df['mean'].round(4).astype(str) + '+-' +(1.96 * result_df['std']/ 5**(1/2)).round(4).astype(str)
    result_df.pivot(index = ['data', 'target'], values = 'view', columns = 'model').to_csv("clarified_results.csv")
    result_df.to_csv("result.csv")

if __name__ == '__main__':
    run_num_past_test()
    run_curr_step_test()