from torch.utils.data import Dataset
import torch as tr
import numpy as np
import os


def make_dataset(base_dir, exp, input_info, start_data_ind, end_data_ind):
    data = {}
    for data_ind in range(start_data_ind, end_data_ind):
        print('MAKE DATA NUM, ', data_ind)
        data_dir = 'agent_30_id_' + str(data_ind)
        data_dir = os.path.join(base_dir, data_dir)
        # Load questions
        c_board = np.load(data_dir + '/curr_board.npz')
        c_src = np.load(data_dir + "/curr_src.npz")
        c_send = np.load(data_dir + "/curr_send.npz")

        data['c_board'] = c_board['total'].astype(np.uint8)
        data["c_src"] = c_src['total'].astype(np.uint8)
        data["c_send"] = c_send['total'].astype(np.uint8)

        # Load labels
        t_order = np.load(data_dir + "/target_order.npz")
        t_dst = np.load(data_dir + "/target_dst.npz")
        t_recv = np.load(data_dir + "/target_recv.npz")
        t_true_prob = np.load(data_dir + "/target_true_prob.npz")

        data["t_order"] = t_order['total'].astype(np.uint8)
        data['t_order'] = np.argmax(data['t_order'], axis=-1)
        data["t_dst"] = t_dst['total'].astype(np.uint8)
        data["t_dst"] = np.argmax(data['t_dst'], axis=-1)
        data["t_recv"] = t_recv['total'].astype(np.uint8)
        data["t_true_prob"] = t_true_prob['total'].astype(np.float16)

        if input_info["agent_info"]:
            me_weights = np.load(data_dir + '/me_weights.npz')
            other_weights = np.load(data_dir + '/other_weights.npz')
            data['me_weights'] = me_weights['total']
            data['other_weights'] = other_weights['total']

            me_index = np.load(data_dir + '/me_index.npz')
            other_index = np.load(data_dir + '/other_index.npz')
            data['me_index'] = me_index['total'].astype(np.uint8)
            data['other_index'] = other_index['total'].astype(np.uint8)
        if input_info["use_past"]:
            p_order = np.load(data_dir + "/past_order.npz")
            p_message = np.load(data_dir + '/past_message.npz')
            p_board = np.load(data_dir + '/past_board.npz')
            data["p_order"] = p_order['total'].astype(np.float16)
            data['p_message'] = p_message['total'].astype(np.float16)
            data['p_board'] = p_board['total'].astype(np.float16)
        if input_info["use_cur"]:
            c_message = np.load(data_dir + '/curr_message.npz')
            c_order = np.load(data_dir + "/curr_order.npz")
            data['c_message'] = c_message['total'].astype(np.float16)
            data["c_order"] = c_order['total'].astype(np.float16)

        if input_info["use_internal"]:
            c_other_internal = np.load(data_dir + '/curr_other_internal.npz')
            c_other_map_tactician = np.load(data_dir + '/curr_other_map_tactician.npz')
            data['c_other_internal'] = c_other_internal['total']
            data['c_other_map_tactician'] = c_other_map_tactician['total']

        if data_ind == start_data_ind:
            tom_train_dataset = ToMDataset(data, exp, input_info)
        else:
            tom_train_dataset.add(data)
            dk = list(data.keys())
            for k in dk:
                del data[k]
            del data
            data = {}

    return tom_train_dataset


class ToMDataset(Dataset):

    def __init__(self, data, exp, input_info):
        self.exp = exp
        self.input_info = input_info
        self.input_data = dict()

        for k, v in data.items():
            self.input_data[k] = tr.from_numpy(v)

        dk = list(data.keys())
        for k in dk:
            del data[k]
        del data

    def add(self, data):
        dk = data.keys()
        for k in dk:
            self.input_data[k] = tr.cat([self.input_data[k],  tr.from_numpy(data[k])], dim=0)

        dk = list(data.keys())
        for k in dk:
            del data[k]
        del data

    def __len__(self):
        return len(self.input_data["c_src"])

    def __getitem__(self, ind):
        # for k in self.input_data.keys():
        #     print(k, self.input_data[k].shape)
        if self.exp == "oracle":
            return self.input_data["c_board"][ind], self.input_data["c_src"][ind], self.input_data["c_message"][ind], \
                   self.input_data["t_order"][ind], self.input_data["t_dst"][ind], self.input_data["t_recv"][ind], \
                   self.input_data["me_weights"][ind], self.input_data["other_weights"][ind], self.input_data["me_index"][ind], \
                   self.input_data["other_index"][ind], \
                   self.input_data["c_send"][ind], self.input_data["c_order"][ind], self.input_data["c_other_internal"][ind], \
                   self.input_data["t_true_prob"][ind]
        if self.exp == "only_lstm":
            return self.input_data["c_board"][ind], self.input_data["c_src"][ind], self.input_data["c_message"][ind], \
                   self.input_data["t_order"][ind], self.input_data["t_dst"][ind], self.input_data["t_recv"][ind], \
                   self.input_data["me_weights"][ind], self.input_data["other_weights"][ind], self.input_data["me_index"][ind], \
                   self.input_data["other_index"][ind], self.input_data["c_send"][ind], self.input_data["c_order"][ind], \
                   self.input_data["t_true_prob"][ind]
        if self.exp == "tomnet":
            return self.input_data["p_board"][ind], self.input_data["p_order"][ind], self.input_data["p_message"][ind],\
                   self.input_data["c_board"][ind], self.input_data["c_src"][ind], self.input_data["c_message"][ind], \
                   self.input_data["t_order"][ind], self.input_data["t_dst"][ind], self.input_data["t_recv"][ind], \
                   self.input_data["me_weights"][ind], self.input_data["other_weights"][ind], self.input_data["me_index"][ind], \
                   self.input_data["other_index"][ind], self.input_data["c_send"][ind], self.input_data["c_order"][ind], \
                   self.input_data["t_true_prob"][ind]
        if self.exp == "attention":
            return self.input_data["p_board"][ind], self.input_data["p_order"][ind], self.input_data["p_message"][ind], \
                   self.input_data["c_board"][ind], self.input_data["c_src"][ind], self.input_data["c_message"][ind], \
                   self.input_data["t_order"][ind], self.input_data["t_dst"][ind], self.input_data["t_recv"][ind], \
                   self.input_data["me_weights"][ind], self.input_data["other_weights"][ind], self.input_data["me_index"][ind], \
                   self.input_data["other_index"][ind], self.input_data["c_send"][ind], self.input_data["c_order"][ind], \
                   self.input_data["t_true_prob"][ind]