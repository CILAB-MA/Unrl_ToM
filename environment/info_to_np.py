import numpy as np
from env import PressDiplomacyEnv
import os, sys, copy
from utils import get_adj_matrix
sys.path.append(os.getcwd())

def static_dict_to_np(static_dict, map):
    static_dict_adj = np.zeros((len(static_dict['locs']), len(static_dict['locs'])), dtype=np.int8)
    static_dict_others = np.zeros((len(static_dict['locs']), 2))
    adj = get_adj_matrix(static_dict['locs'], map)
    static_dict_adj[:, :] = adj
    #print(static_dict['locs'])
    scs = get_scs_matrix(static_dict['locs'], static_dict['scs'])

    static_dict_others[:, 0] = scs
    area_type = get_area_matrix(static_dict['locs'], static_dict['area_type'])
    static_dict_others[:, 1] = area_type

    return static_dict_adj, static_dict_others

def get_scs_matrix(locs, scs):
    dummy_zeros = np.zeros((len(locs)), dtype=np.int8)
    for sc in scs:
        loc_ind = locs.index(sc)
        dummy_zeros[loc_ind] = 1
    return dummy_zeros

def get_area_matrix(locs, area_type):
    dummy_zeros = np.zeros((len(locs)), dtype=np.int8)
    areas = ['COAST', 'WATER', 'LAND']
    for i, loc in enumerate(locs):
        ind = areas.index(area_type[loc])
        dummy_zeros[i] = ind
    return dummy_zeros

if __name__ == '__main__':
    env = PressDiplomacyEnv()
    env.reset()
    map = copy.deepcopy(env.map)
    static_infos = env.static_infos
    static_dict_to_np(static_infos, map)