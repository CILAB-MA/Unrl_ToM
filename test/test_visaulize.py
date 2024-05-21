import copy, os, sys, yaml
sys.path.append(os.getcwd())
import pandas as pd
import torch.nn as nn
import torch as tr
from utils.visualize import Visualizer
from torch.utils.data import DataLoader
from experiment_pred_oracle.PredNet import FC_PredNet as OracleNet
from experiment_hc.PredNet import FC_PredNet as AttentionNet
from experiment_only.PredNet import Only_lstm_PredNet
from utils import dataset
from utils.tools import *

BASE_DIR = '/workspace/uwonsang/data/'

DATA = ['test', 'betray', 'complex','dummy']

EXPERIMENT = {
    'attention' : {'config' : './configs/config.yaml',
                   'network' : 'AttentionNet',
                   'saved_dir' : './attention/',
                }
}

def run_visualize():
    device = 'cuda' if tr.cuda.is_available() else 'cpu'
    visualizer = Visualizer()

    for model in EXPERIMENT.keys():
        with open(EXPERIMENT[model]['config']) as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            configs.update({'charnet_type': 'fc', 'no_char': False})
        saved_dir = EXPERIMENT[model]['saved_dir']
        net = eval(EXPERIMENT[model]['network'])(device=device, configs=configs, norm_adjacency=norm_adjacency)
        net.to('cuda')
    for saved_model in os.listdir(EXPERIMENT[model]['saved_dir']):
        net.load_state_dict(tr.load(saved_dir + saved_model))

        train_e_chars_list = []
        train_me_weights_list = []
        train_other_weights_list = []
        tom_dataset = dataset.make_dataset(BASE_DIR + f'test_data/', configs['exp'], 0, 1)
        dataloader = DataLoader(tom_dataset, batch_size=128, shuffle=False)
        net.eval()

        train_e_chars_list = []
        train_me_weights_list = []
        train_other_weights_list = []
        train_other_powers_list = []

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

            if model == 'attention':
                pred_order, pred_dst, pred_response, e_char, weights = net(p_board, p_order, p_message, c_board, c_src,
                                                                           c_message, c_send, c_order)
            else:
                raise Exception('invalid model')
            e_char_np = e_char.cpu().detach().numpy()
            train_other_powers_list.append(other_ind[:,-1].cpu().detach().numpy())
            train_e_chars_list.append(e_char_np)
            train_me_weights_list.append(me_weights.cpu().detach().squeeze(-1).numpy())
            train_other_weights_list.append(other_weights.cpu().detach().squeeze(-1).numpy())

        pil_image_me = visualizer.get_char(train_e_chars_list, 'test', train_me_weights_list)
        pil_image_powers = visualizer.get_power_char(train_e_chars_list, 'test', train_other_powers_list,
                                                     train_other_weights_list, saved_model)
        pil_image_other = visualizer.get_char(train_e_chars_list, 'test', train_other_weights_list)

        pil_image_me.save(f'SAVE_{saved_model}_ME_CHAR.jpg', dpi=300)
        pil_image_other.save(f'SAVE_{saved_model}_OTHER_CHAR.jpg', dpi=300)
