import torch.nn.functional as F
import torch as tr
import torch.nn as nn


class FC_PredNet(nn.Module):
    def __init__(self, device, norm_adjacency, configs):
        super(FC_PredNet, self).__init__()

        if configs['no_char']:
            self.hidden_size = 32 + 128 + 128 + 3
        else:
            self.hidden_size = configs['char_output_dim'] + 32 + 128 + 128 + 3  # 3 is internal state
        self.past_board_fc1 = nn.Linear(configs['num_loc'] * configs['board_feat'], 1024)
        self.past_board_fc2 = nn.Linear(1024, 128)

        self.past_order_fc1 = nn.Linear(configs['num_order'] * configs['order_feat'], 1024)
        self.past_order_fc2 = nn.Linear(1024, 128)

        self.past_msg_fc1 = nn.Linear(2 * configs['msg_feat'], 64)
        self.past_msg_fc2 = nn.Linear(64, 32)

        self.norm_adjacency = norm_adjacency
        self.relu = nn.ReLU(inplace=True)

        # self.curr_fc1 = nn.Linear(49 * configs['lstm_hidden_dim'], configs['lstm_hidden_dim'])
        self.query_fc1 = nn.Linear(configs['num_loc'] + configs['send_dim'] + (configs['num_loc'] * configs['board_feat']),
                                   1024)
        self.query_fc2 = nn.Linear(1024, configs['query_output_dim'])

        if configs['no_char']:
            dim_feature = configs['lstm_hidden_dim'] + configs['query_output_dim']
        else:
            dim_feature = configs['lstm_hidden_dim'] + configs['query_output_dim'] + configs['char_output_dim'] + (2 * configs['num_agent'])

        self.fc1 = nn.Linear(dim_feature, 1024)  # 49 = curr epi num step
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

    def forward(self, curr_board, curr_src, curr_msg, curr_send, curr_order, w_me, w_other, me_ind, other_ind, int_other):

        curr_board_last = curr_board[:, -1, :]
        prep_curr_board = curr_board[:, : -1, :]  # (batch, 49, 81, 35)
        prep_curr_msg = curr_msg[:, :-1, :]  # (batch, 49, 90)
        prep_curr_internal = int_other[:, :-1, :]  # (batch, 49, 3)

        curr_me_ind = F.one_hot(me_ind[:, -1].to(tr.int64), num_classes=self.configs['num_agent'])
        curr_me_ind = curr_me_ind.reshape(-1, self.configs['num_agent']).unsqueeze(1)
        curr_other_ind = F.one_hot(other_ind[:, -1].to(tr.int64), num_classes=self.configs['num_agent'])
        curr_other_ind = curr_other_ind.reshape(-1, self.configs['num_agent']).unsqueeze(1)

        batch, curr_num_step, num_loc, num_board_feature = prep_curr_board.shape
        _, curr_num_msg, _ = prep_curr_msg.shape

        hx, cx = self.init_hidden(batch)

        e_char_2d = tr.cat([w_other.to(self.device), w_me.to(self.device)], dim=-1)

        e_char = e_char_2d.unsqueeze(1)
        e_char = e_char.repeat(1, 1, 8)
        e_char_repeat = e_char.repeat(1, curr_num_step, 1)

        prep_curr_msg = prep_curr_msg.reshape(batch, curr_num_msg, -1)
        prep_curr_msg = self.relu(self.past_msg_fc1(prep_curr_msg))
        prep_curr_msg = self.relu(self.past_msg_fc2(prep_curr_msg))

        prep_curr_board = prep_curr_board.reshape(batch, curr_num_step, -1)
        prep_curr_board = self.relu(self.past_board_fc1(prep_curr_board))
        prep_curr_board = self.relu(self.past_board_fc2(prep_curr_board))

        curr_order = self.relu(self.past_order_fc1(curr_order))  # (batch, 49, 181) -> (batch, 1024)
        curr_order = self.relu(self.past_order_fc2(curr_order))

        x_concat = tr.cat([e_char_repeat, prep_curr_msg, prep_curr_board, curr_order, prep_curr_internal], dim=-1)
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

        # print(final_out_last.shape, curr_other_ind.shape)
        if self.no_char:
            x = tr.cat([final_out_last, q], dim=-1)
        else:
            # print(final_out_last.shape, q.shape, curr_other_ind.shape, curr_me_ind.shape)
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
            return order, dst, response, e_char_2d
