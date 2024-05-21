from torch.utils.data import Dataset
import torch as tr
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser('For ToM Passive Exp')
    parser.add_argument('--data_path', type=str, default='/app/data/agent_30_id_13')
    return parser.parse_args()

def load_dataset(args):

    # data["past_order"] = np.load(data_dir + "/past_order.npy")
    # data['past_other_internal'] = np.load(data_dir + '/past_other_internal.npy')
    # data['past_other_map_tactician'] = np.load(data_dir + '/past_other_map_tactician.npy')
    # data['past_me_map_tactician'] = np.load(data_dir + '/past_me_map_tactician.npy')
    # data['past_me_internal'] = np.load(data_dir + '/past_me_internal.npy')
    # data['past_message'] = np.load(data_dir + '/past_message.npy')
    # data['past_board'] = np.load(data_dir + '/past_board.npy')
    #
    # data["curr_src"] = np.load(data_dir + "/curr_src.npy")
    # data['curr_other_internal'] = np.load(data_dir + '/curr_other_internal.npy')
    # data['curr_other_map_tactician'] = np.load(data_dir + '/curr_other_map_tactician.npy')
    # data['curr_me_map_tactician'] = np.load(data_dir + '/curr_me_map_tactician.npy')
    # data['curr_me_internal'] = np.load(data_dir + '/curr_me_internal.npy')
    # data['curr_message'] = np.load(data_dir + '/curr_message.npy')
    # data['curr_board'] = np.load(data_dir + '/curr_board.npy')
    # data['curr_order'] = np.load(data_dir + '/curr_order.npy')
    # data["curr_send"] = np.load(data_dir + "/curr_send.npy")
    #
    # data['me_weights'] = np.load(data_dir + '/me_weights.npy')
    # data['other_weights'] = np.load(data_dir + '/other_weights.npy')
    # data['me_index'] = np.load(data_dir + '/me_index.npy')
    # data['other_index'] = np.load(data_dir + '/other_index.npy')
    # data["target_order"] = np.load(data_dir + "/target_order.npy")
    # data["target_dst"] = np.load(data_dir + "/target_dst.npy")
    # data["target_recv"] = np.load(data_dir + "/target_recv.npy")
    print('=============load_data===============================')
    print("data_path : ", args.data_path)
    data = np.load(args.data_path + '/past_message.npy')
    print('data_shape : ', data.shape)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    load_dataset(args)