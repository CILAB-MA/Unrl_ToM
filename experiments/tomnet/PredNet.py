from .CharNet import *
import torch.nn.functional as F
import torch.nn as nn

class Only_lstm_PredNet(nn.Module):
    def __init__(self, device, norm_adjacency, configs):
        super(Only_lstm_PredNet, self).__init__()

        if configs['no_char']:
            self.hidden_size = (configs['char_output_dim'] + configs['num_msg'] * configs['msg_feat']) + (128)
        else:
            self.hidden_size = (configs['char_output_dim'] + 32) + (128) + (128)

        self.past_board_fc1 = nn.Linear(configs['num_loc'] * configs['board_feat'], 1024)
        self.past_board_fc2 = nn.Linear(1024, 128)

        self.past_order_fc1 = nn.Linear(configs['num_order'] * configs['order_feat'], 1024)
        self.past_order_fc2 = nn.Linear(1024, 128)

        self.past_msg_fc1 = nn.Linear(2 * configs['msg_feat'], 64)
        self.past_msg_fc2 = nn.Linear(64, 32)

        self.norm_adjacency = norm_adjacency

        if configs['gcn']:
            self.e_char = GCN_CharNet(self.norm_adjacency, device, configs)
        else:
            self.e_char = FC_CharNet(device, configs)

        self.relu = nn.ReLU(inplace=True)

        self.query_fc1 = nn.Linear(configs['num_loc'] + configs['send_dim'] + (configs['num_loc'] * configs['board_feat']),
                                   1024)
        self.query_fc2 = nn.Linear(1024, configs['query_output_dim'])

        if configs['no_char']:
            dim_feature = configs['lstm_hidden_dim'] + configs['query_output_dim']
        else:
            dim_feature = configs['lstm_hidden_dim'] + configs['query_output_dim'] + configs['char_output_dim'] + \
                          (2 * configs['num_agent'])

        self.fc1 = nn.Linear(dim_feature, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.device = device
        self.no_char = configs['no_char']
        self.softmax = nn.Softmax(dim=-1)
        self.lstm = nn.LSTMCell(self.hidden_size, configs['lstm_hidden_dim'])
        self.lstm_hidden = configs['lstm_hidden_dim']
        self.device = device

        self.order_head = nn.Sequential(
                                        nn.Linear(128, 4),
                                        nn.LogSoftmax(dim=-1))

        self.dst_head = nn.Sequential(
                                      nn.Linear(128, configs['num_loc']),
                                      nn.LogSoftmax(dim=-1))

        self.response_head = nn.Sequential(
                                           nn.Linear(128, 2),
                                           nn.LogSoftmax(dim=-1))
        self.configs = configs

    def init_hidden(self, batch_size):
        return (tr.randn(batch_size, self.lstm_hidden, device=self.device),
                tr.randn(batch_size, self.lstm_hidden, device=self.device))

    def forward(self, past_board, past_order, past_msg, curr_board, curr_src, curr_msg, curr_send, curr_order, other_ind, me_ind):

        curr_board_last = curr_board[:, -1, :]
        prep_curr_board = curr_board[:, : -1, :]
        prep_curr_msg = curr_msg[:, :-1, :]

        curr_me_ind = F.one_hot(me_ind[:, -1].to(tr.int64), num_classes=self.configs['num_agent'])
        curr_me_ind = curr_me_ind.reshape(-1, self.configs['num_agent']).unsqueeze(1)
        curr_other_ind = F.one_hot(other_ind[:, -1].to(tr.int64), num_classes=self.configs['num_agent'])
        curr_other_ind = curr_other_ind.reshape(-1, self.configs['num_agent']).unsqueeze(1)
        past_me_ind = F.one_hot(me_ind[:, :-1].to(tr.int64), num_classes=self.configs['num_agent'])
        past_me_ind = past_me_ind.reshape(-1, 4, self.configs['num_agent'])
        past_other_ind = F.one_hot(other_ind[:, :-1].to(tr.int64), num_classes=self.configs['num_agent'])
        past_other_ind = past_other_ind.reshape(-1, 4, self.configs['num_agent'])

        batch, curr_num_step, num_loc, num_board_feature = prep_curr_board.shape
        _, curr_num_msg, _ = prep_curr_msg.shape
        _, _, s, _, _ = past_board.shape

        hx, cx = self.init_hidden(batch)

        # char_net
        if s == 0:
            e_char_2d = tr.zeros((batch, curr_num_step, self.configs['char_output_dim']), device=self.device)
        else:
            e_char_2d, e_char_total = self.e_char(past_board, past_order, past_msg, past_other_ind, past_me_ind)

        e_char = e_char_2d.unsqueeze(1)
        e_char_repeat = e_char.repeat(1, curr_num_step, 1)

        prep_curr_msg = prep_curr_msg.reshape(batch, curr_num_msg, -1)
        prep_curr_msg = self.relu(self.past_msg_fc1(prep_curr_msg))
        prep_curr_msg = self.relu(self.past_msg_fc2(prep_curr_msg))

        board = prep_curr_board.reshape(batch, curr_num_step, -1)
        board = self.relu(self.past_board_fc1(board))
        board = self.relu(self.past_board_fc2(board))

        curr_order = self.relu(self.past_order_fc1(curr_order))
        curr_order = self.relu(self.past_order_fc2(curr_order))

        x_concat = tr.cat([e_char_repeat, board, prep_curr_msg, curr_order], dim=-1)
        x_concat_out = x_concat.transpose(0, 1)  # (batch, curr_num_step, -1) -> (curr_num_step, batch, -1)

        # mental_net
        outs = []
        for step in range(curr_num_step):
            hx, cx = self.lstm(x_concat_out[step], (hx, cx))
            outs.append(hx)
        final_out = tr.stack(outs, dim=1)
        final_out_last = final_out[:, -1:, :]
        curr_board_last = curr_board_last.reshape(batch, 1, -1)

        # embed query state
        query = tr.cat([curr_src.reshape(batch, 1, -1), curr_send.reshape(batch, 1, -1), curr_board_last], dim=-1)
        query = query.reshape(batch, -1)
        q = self.relu(self.query_fc1(query))
        q = self.relu(self.query_fc2(q))

        if self.no_char:
            x = tr.cat([final_out_last, q], dim=-1)
        else:
            x = tr.cat([final_out_last, e_char, q.reshape(batch, 1, -1), curr_other_ind, curr_me_ind], dim=-1)
        x = x.reshape(batch, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        order = self.order_head(x)
        dst = self.dst_head(x)
        response = self.response_head(x)

        if self.no_char:
            return order, dst, response, None
        else:
            return order, dst, response, e_char_2d, e_char_total
