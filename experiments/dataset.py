from torch.utils.data import Dataset
import torch as tr
import numpy as np
import os


def make_dataset(base_dir, exp, input_info, start_data_ind, end_data_ind):
    data = {}

    for data_ind in range(start_data_ind, end_data_ind):
        data_dir = 'agent_30_id_' + str(data_ind)
        data_dir = os.path.join(base_dir, data_dir)
        # Load questions
        data['c_board'] = np.load(data_dir + '/curr_board.npy').astype(np.uint8)
        data["c_src"] = np.load(data_dir + "/curr_src.npy").astype(np.uint8)
        data["c_send"] = np.load(data_dir + "/curr_send.npy").astype(np.uint8).reshape(-1, 1, 43)

        # Load labels
        data["t_order"] = np.load(data_dir + "/target_order.npy").astype(np.uint8)
        data["t_order"] = np.argmax(data["t_order"], axis=-1)
        data["t_dst"] = np.load(data_dir + "/target_dst.npy").astype(np.uint8)
        data["t_dst"] = np.argmax(data["t_dst"], axis=-1)
        data["t_recv"] = np.load(data_dir + "/target_recv.npy").astype(np.uint8)

        if input_info["agent_info"]:
            data['me_weights'] = np.load(data_dir + '/me_weights.npy')
            data['other_weights'] = np.load(data_dir + '/other_weights.npy')
            data['me_index'] = np.load(data_dir + '/me_index.npy').astype(np.uint8)
            data['other_index'] = np.load(data_dir + '/other_index.npy').astype(np.uint8)
        if input_info["use_past"]:
            data["p_order"] = np.load(data_dir + "/past_order.npy").astype(np.uint8)
            data['p_message'] = np.load(data_dir + '/past_message.npy').astype(np.uint8)
            data['p_board'] = np.load(data_dir + '/past_board.npy').astype(np.uint8)
        if input_info["use_cur"]:
            data['c_message'] = np.load(data_dir + '/curr_message.npy').astype(np.uint8)
            data["c_order"] = np.load(data_dir + "/curr_order.npy").astype(np.uint8)
        if input_info["use_internal"]:
            data['c_other_internal'] = np.load(data_dir + '/curr_other_internal.npy')
            data['c_other_map_tactician'] = np.load(data_dir + '/curr_other_map_tactician.npy')

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
        if self.exp == "oracle":
            return self.input_data["c_board"][ind], self.input_data["c_src"][ind], self.input_data["c_message"][ind], \
                   self.input_data["t_order"][ind], self.input_data["t_dst"][ind], self.input_data["t_recv"][ind], \
                   self.input_data["me_weights"][ind], self.input_data["other_weights"][ind], self.input_data["me_index"][ind], \
                   self.input_data["other_index"][ind], \
                   self.input_data["c_send"][ind], self.input_data["c_order"][ind], self.input_data["c_other_internal"][ind]
        if self.exp == "only_lstm":
            return self.input_data["c_board"][ind], self.input_data["c_src"][ind], self.input_data["c_message"][ind], \
                   self.input_data["t_order"][ind], self.input_data["t_dst"][ind], self.input_data["t_recv"][ind], \
                   self.input_data["me_weights"][ind], self.input_data["other_weights"][ind], self.input_data["me_index"][ind], \
                   self.input_data["other_index"][ind], self.input_data["c_send"][ind], self.input_data["c_order"][ind]
        if self.exp == "tomnet":
            return self.input_data["p_board"][ind], self.input_data["p_order"][ind], self.input_data["p_message"][ind],\
                   self.input_data["c_board"][ind], self.input_data["c_src"][ind], self.input_data["c_message"][ind], \
                   self.input_data["t_order"][ind], self.input_data["t_dst"][ind], self.input_data["t_recv"][ind], \
                   self.input_data["me_weights"][ind], self.input_data["other_weights"][ind], self.input_data["me_index"][ind], \
                   self.input_data["other_index"][ind], self.input_data["c_send"][ind], self.input_data["c_order"][ind]
        if self.exp == "attention":
            return self.input_data["p_board"][ind], self.input_data["p_order"][ind], self.input_data["p_message"][ind], \
                   self.input_data["c_board"][ind], self.input_data["c_src"][ind], self.input_data["c_message"][ind], \
                   self.input_data["t_order"][ind], self.input_data["t_dst"][ind], self.input_data["t_recv"][ind], \
                   self.input_data["me_weights"][ind], self.input_data["other_weights"][ind], self.input_data["me_index"][ind], \
                   self.input_data["other_index"][ind], self.input_data["c_send"][ind], self.input_data["c_order"][ind]


