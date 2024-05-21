import torch.nn.functional as F
import torch as tr
import torch.nn as nn

add_order = True
me_char = True
no_message_pred = False
only_lstm = True
add_power = True
add_internal = False
masking = False


class FC_PredNet(nn.Module):
    def __init__(self, device, norm_adjacency, configs):
        super(FC_PredNet, self).__init__()

        if only_lstm:
            self.hidden_size =2 * configs['msg_feat'] + configs['num_board'] * configs['board_feat']
            if add_order:
                self.hidden_size += configs['num_order'] * configs['order_feat']
            if add_internal:
                self.hidden_size += configs['num_internal'] * configs['internal_feat']

        else:
            self.hidden_size = configs['message_output_dim'] + configs['board_output_dim']
            if add_order:
                self.hidden_size += configs['order_output_dim']
            if add_internal:
                self.hidden_size += configs["internal_dim"]

        self.norm_adjacency = norm_adjacency

        self.relu = nn.ReLU(inplace=True)
        self.fc_msg1 = nn.Linear(2 * configs['msg_feat'], 1024)
        self.fc_msg2 = nn.Linear(1024, 128)
        self.fc_msg3 = nn.Linear(128, configs['message_output_dim'])
        if configs['no_char']:
            self.fc_board1 = nn.Linear(configs['num_board'] * configs['board_feat'], 1024)
            self.fc_board2 = nn.Linear(1024, 128)
            self.fc_board3 = nn.Linear(128, configs['board_output_dim'])
        else:
            self.fc_board1 = nn.Linear(configs['num_board'] * (configs['board_feat']), 1024)
            self.fc_board2 = nn.Linear(1024, 128)
            self.fc_board3 = nn.Linear(128, configs['board_output_dim'])

        if add_order:
            self.fc_order1 = nn.Linear(configs['num_order'] * (configs['order_feat']), 1024)
            self.fc_order2 = nn.Linear(1024, 128)
            self.fc_order3 = nn.Linear(128, configs['order_output_dim'])

        dim_feature = self.hidden_size + configs['src_dim'] + configs['send_dim'] + configs['char_output_dim'] + \
                      configs['num_board'] * configs['board_feat']

        if me_char:
            dim_feature += 2
        if add_power:
            dim_feature += 14

        self.fc1 = nn.Linear(dim_feature, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.device = device
        self.no_char = configs['no_char']
        self.softmax = nn.Softmax(dim=-1)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.device = device
        self.order_head = nn.Sequential(nn.ReLU(),
                                        nn.Linear(256, 4),
                                        nn.LogSoftmax(dim=-1))
        if masking:
            self.dst_head = nn.Sequential(nn.ReLU(),
                                          nn.Linear(256, 81))
            self.log_softmax = nn.LogSoftmax(dim=-1)
        else:
            self.dst_head = nn.Sequential(nn.ReLU(),
                                          nn.Linear(256, 81),
                                          nn.LogSoftmax(dim=-1))
        self.response_head = nn.Sequential(nn.ReLU(),
                                      nn.Linear(256, 2),
                                      nn.LogSoftmax(dim=-1))
        self.configs = configs

    def init_hidden(self, batch_size):
        return (tr.randn(batch_size, self.hidden_size, device=self.device),
                tr.randn(batch_size, self.hidden_size, device=self.device))

    def forward(self, curr_board, curr_src, curr_msg, curr_send, curr_order, w_me, w_other, i_me, i_other, int_other):
        curr_board_last = curr_board[:, -1, :]
        prep_curr_order = curr_order[:, :, :]
        prep_curr_board = curr_board[:, :-1, :]
        prep_curr_msg = curr_msg[:, :-1, :]
        prep_int_other = int_other[:, :-1, :]

        _, curr_num_order, _ = prep_curr_order.shape
        batch, curr_num_step, num_loc, num_board_feature = prep_curr_board.shape
        _, curr_num_msg, _ = prep_curr_msg.shape
        _, curr_num_int, num_int_feature = prep_int_other.shape

        hx, cx = self.init_hidden(batch)

        device = 'cuda' if tr.cuda.is_available() else 'cpu'
        if me_char:
            e_char_2d = tr.cat([w_other.to(device), w_me.to(device)], dim=-1)
        else:
            e_char_2d = w_other.to(device)

        e_char_2d = e_char_2d.unsqueeze(1)

        if add_power:
            i_me = F.one_hot(i_me[:, -1].to(tr.int64), num_classes=7).float().cuda()
            i_me = i_me.reshape(batch, -1).unsqueeze(1)
            i_other = F.one_hot(i_other[:, -1].to(tr.int64), num_classes=7).float().cuda()
            i_other = i_other.reshape(batch, -1).unsqueeze(1)

        if only_lstm:
            if add_order:
                order_feat = prep_curr_order.reshape(batch, curr_num_order, -1)
            obs_feature = prep_curr_board.reshape(batch, curr_num_step, -1)
            message_feat = prep_curr_msg.reshape(batch, curr_num_msg, -1)
            int_feature = prep_int_other.reshape(batch, curr_num_int, -1)
        else:
            if add_order:
                order = prep_curr_order.reshape(batch * curr_num_order, -1).to(device)
                order_feature = self.relu(self.fc_order1(order))
                order_feature = self.relu(self.fc_order2(order_feature))
                order_feature = self.relu(self.fc_order3(order_feature))
                order_feat = order_feature.view(batch, curr_num_order, -1)

            board = prep_curr_board.reshape(batch * curr_num_step, -1)
            board_feature = self.relu(self.fc_board1(board))
            board_feature = self.relu(self.fc_board2(board_feature))
            board_feature = self.relu(self.fc_board3(board_feature))
            obs_feature = board_feature.view(batch, curr_num_step, -1)

            prep_curr_msg = prep_curr_msg.reshape(batch * curr_num_msg, -1)
            message_feat = self.relu(self.fc_msg1(prep_curr_msg))
            message_feat = self.relu(self.fc_msg2(message_feat))
            message_feat = self.relu(self.fc_msg3(message_feat))
            message_feat = message_feat.view(batch, curr_num_msg, -1)

        if add_order and not add_internal:
            x_concat = tr.cat([order_feat, obs_feature, message_feat], dim=-1)
        elif add_order and add_internal:
            x_concat = tr.cat([order_feat, obs_feature, message_feat, int_feature], dim=-1)
        elif not add_order and add_internal:
            x_concat = tr.cat([obs_feature, message_feat, int_feature], dim=-1)
        else:
            x_concat = tr.cat([obs_feature, message_feat], dim=-1)
        x_concat = x_concat.reshape(batch, -1)

        outs = []
        for step in range(curr_num_step):
            hx, cx = self.lstm(x_concat.reshape(curr_num_step, batch, -1)[step], (hx, cx))
            outs.append(hx)
        final_out = tr.stack(outs, dim=1)
        final_out_last = final_out[:, -1:, :]
        curr_board_last = curr_board_last.reshape(batch, 1, -1)
        if add_power:
            x = tr.cat([final_out_last, e_char_2d, i_me, i_other, curr_src, curr_send, curr_board_last], dim=-1)
        else:
            x = tr.cat([final_out_last, e_char_2d, curr_src, curr_send, curr_board_last], dim=-1)
        x = x.reshape(batch, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        order = self.order_head(x)
        dst = self.dst_head(x)
        e_char_2d = e_char_2d.squeeze(1)

        if masking:
            curr_src = tr.argmax(curr_src.squeeze(1).to(tr.int64), dim=1)
            norm_adjacency = tr.from_numpy(self.norm_adjacency)
            masking_matrix = norm_adjacency.to(device)[curr_src]
            dst = dst * masking_matrix
            dst = self.log_softmax(dst)

        if no_message_pred:
            return order, dst, 0, e_char_2d
        else:
            response = self.response_head(x)
            return order, dst, response, e_char_2d


