import os
import numpy as np
import datetime as dt
import torch
from dateutil.tz import gettz

from diplomacy import Map

### num 81 지역명 확인하기
STANDARD_TOPO_LOCS = ['YOR', 'EDI', 'LON', 'LVP', 'NTH', 'WAL', 'CLY',
                      'NWG', 'ENG', 'IRI', 'NAO', 'BEL', 'DEN', 'HEL',
                      'HOL', 'NWY', 'SKA', 'BAR', 'BRE', 'MAO', 'PIC',
                      'BUR', 'RUH', 'BAL', 'KIE', 'SWE', 'FIN', 'STP',
                      'STP/NC', 'GAS', 'PAR', 'NAF', 'POR', 'SPA', 'SPA/NC',
                      'SPA/SC', 'WES', 'MAR', 'MUN', 'BER', 'BOT', 'LVN',
                      'PRU', 'STP/SC', 'MOS', 'TUN', 'LYO', 'TYS', 'PIE',
                      'BOH', 'SIL', 'TYR', 'WAR', 'SEV', 'UKR', 'ION',
                      'TUS', 'NAP', 'ROM', 'VEN', 'GAL', 'VIE', 'TRI',
                      'ARM', 'BLA', 'RUM', 'ADR', 'AEG', 'ALB', 'APU',
                      'EAS', 'GRE', 'BUD', 'SER', 'ANK', 'SMY', 'SYR',
                      'BUL', 'BUL/EC', 'CON', 'BUL/SC']
ADJACENCY_MATRIX = {}
SORTED_LOCS = {}


def get_sorted_locs(map_object, map_name):

    if map_object.name not in SORTED_LOCS:
        key = None if not map_object.name.startswith(map_name) else STANDARD_TOPO_LOCS.index
        locs = [l.upper() for l in map_object.locs if map_object.area_type(l) != 'SHUT']
        SORTED_LOCS[map_object.name] = sorted(locs, key=key)
    return SORTED_LOCS[map_object.name]


def get_adjacency_matrix(map_name='standard'):

    if map_name in ADJACENCY_MATRIX:
        return ADJACENCY_MATRIX[map_name]

    current_map = Map(map_name)
    locs = get_sorted_locs(current_map, map_name)
    adjacencies = np.zeros((len(locs), len(locs)), dtype=np.bool)

    for i, loc_1 in enumerate(locs):
        for j, loc_2 in enumerate(locs):
            if current_map.abuts('A', loc_1, '-', loc_2) or current_map.abuts('F', loc_1, '-', loc_2):
                adjacencies[i, j] = 1
            if loc_1 != loc_2 and (loc_1[:3] == loc_2 or loc_1 == loc_2[:3]):
                adjacencies[i, j] = 1

    ADJACENCY_MATRIX[map_name] = adjacencies
    return adjacencies


def preprocess_adjacency(adjacency_matrix):
    """ Symmetrically normalize the adjacency matrix for graph convolutions.
        :param adjacency_matrix: A NxN adjacency matrix
        :return: A normalized NxN adjacency matrix
    """
    # Computing A^~ = A + I_N
    adj = adjacency_matrix
    adj_tilde = adj + np.eye(adj.shape[0])

    # Calculating the sum of each row
    sum_of_row = np.array(adj_tilde.sum(1))

    # Calculating the D tilde matrix ^ (-1/2)
    d_inv_sqrt = np.power(sum_of_row, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # Calculating the normalized adjacency matrix
    norm_adj = adj_tilde.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return np.array(norm_adj, dtype=np.float32)


def film_gcn_res_block(inputs, gamma, beta, gcn_out_dim, norm_adjacency, is_training):

    gcn_input_dim = inputs.shape.as_list(-1) ## input_shape
    assert gcn_input_dim != gcn_out_dim, 'For residual blocks, the in and out dims must be equal'

    gcn_result = GCNLayer(gcn_input_dim, gcn_out_dim, norm_adjacency)(inputs)
    gcn_bn_result = F.batch_norm(gcn_result, training=is_training)
    film_result = gamma * gcn_bn_result + beta
    film_result = F.relu(film_result)

    ## TODO
    ## residual추가하기
    return film_result


def save_model(model, experiment_folder, info):
    model_path = '{}'.format(experiment_folder)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_name = "{}/{}_seed_{}_epoch_{}.pt".format(model_path, info["exp"], info["seed"], info["epoch"])
    torch.save(model.state_dict(), file_name)


def make_folder(num_exp):
    now = dt.datetime.now(gettz('Asia/Seoul'))
    year, month, day, hour, minutes, sec = str(now.year)[-2:], now.month, now.day, now.hour, now.minute, now.second

    foldername = '{}_{}_{}_{}_{}_{}_exp_{}'.format(year, month, day, hour, minutes, sec, num_exp)
    folder_dir = './results/{}'.format(foldername)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    return folder_dir


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    map_name = BASE_DIR + "/small"
    print(map_name)
    adjacency_matrix = get_adjacency_matrix(map_name)
    print(adjacency_matrix.shape)