from .CharNet import *
import torch.nn.functional as F


class FC_PredNet(nn.Module):
    def __init__(self, device, norm_adjacency, configs):
        super(FC_PredNet, self).__init__()
        self.hidden_size = 2 * configs['msg_feat'] + configs['num_board'] * configs['board_feat'] + configs['order_feat']
        self.norm_adjacency = norm_adjacency
        if configs['charnet_type'] == 'fc':
            self.e_char = FC_CharNet(device, configs)
        elif configs['charnet_type'] == 'gcn':
            self.e_char = GCN_CharNet(device, configs, norm_adjacency=self.norm_adjacency)
        self.relu = nn.ReLU(inplace=True)
        self.fc_msg1 = nn.Linear(2 * configs['msg_feat'], 64)
        self.fc_msg2 = nn.Linear(64, 32)
        self.fc_msg3 = nn.Linear(32, configs['message_output_dim'])
        if configs['no_char']:
            self.fc_board1 = nn.Linear(configs['num_board'] * configs['board_feat'], 1024)
            self.fc_board2 = nn.Linear(1024, 128)
            self.fc_board3 = nn.Linear(128, configs['board_output_dim'])
        else:
            self.fc_board1 = nn.Linear(configs['num_board'] * (configs['board_feat']), 1024)
            self.fc_board2 = nn.Linear(1024, 128)
            self.fc_board3 = nn.Linear(128, configs['board_output_dim'])
        self.fc1 = nn.Linear((self.hidden_size + configs['src_dim'] + configs['send_dim'] + configs['char_output_dim'] + (configs['num_board'] * configs['board_feat']) + 14), 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.device = device
        self.no_char = configs['no_char']
        self.softmax = nn.Softmax(dim=-1)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.device = device
        self.order_head = nn.Sequential(nn.ReLU(),
                                        nn.Linear(256, 4),
                                        nn.LogSoftmax(dim=-1))
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

    def forward(self, past_board, past_order, past_msg, curr_board, curr_src, curr_msg, curr_send,
                 curr_order, other_ind, me_ind):

        curr_me_ind = F.one_hot(me_ind[:, -1].to(tr.int64), num_classes=7)
        curr_me_ind = curr_me_ind.reshape(-1, 7).unsqueeze(1)
        curr_other_ind = F.one_hot(other_ind[:, -1].to(tr.int64), num_classes=7)
        curr_other_ind = curr_other_ind.reshape(-1, 7).unsqueeze(1)
        past_me_ind = F.one_hot(me_ind[:, :-1].to(tr.int64), num_classes=7)
        past_me_ind = past_me_ind.reshape(-1, 4, 7)
        past_other_ind = F.one_hot(other_ind[:, :-1].to(tr.int64), num_classes=7)
        past_other_ind = past_other_ind.reshape(-1, 4, 7)
        curr_board_last = curr_board[:, -1, :]
        prep_curr_board = curr_board[:, :  -1, : ]
        prep_curr_msg = curr_msg[:, :-1, :]

        batch, curr_num_step, num_loc, num_board_feature = prep_curr_board.shape
        _, curr_num_msg, _ = prep_curr_msg.shape
        _, _, s, _, _ = past_board.shape

        hx, cx = self.init_hidden(batch)


        if s == 0:
            e_char = tr.zeros((batch, curr_num_step, num_loc, self.configs['char_output_dim']), device=self.device)
        else:
            e_char_2d, weights = self.e_char(past_board, past_order, past_msg, past_other_ind, past_me_ind)
            e_char_2d = e_char_2d.unsqueeze(1)


        prep_curr_msg = prep_curr_msg.reshape(batch, curr_num_msg, -1)
        board = prep_curr_board.reshape(batch, curr_num_step, -1)
        x_concat = tr.cat([board, prep_curr_msg, curr_order], dim=-1)
        x_concat = x_concat.reshape(curr_num_step, batch, -1)
        outs = []
        for step in range(curr_num_step):
            hx, cx = self.lstm(x_concat[step], (hx, cx))
            outs.append(hx)
        final_out = tr.stack(outs, dim=1)
        final_out_last = final_out[:, -1:, :]
        curr_board_last = curr_board_last.reshape(batch, 1, -1)
        x = tr.cat([final_out_last, e_char_2d, curr_src, curr_send, curr_board_last, curr_other_ind, curr_me_ind], dim=-1)
        x = x.reshape(batch, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        order = self.order_head(x)
        dst = self.dst_head(x)
        response = self.response_head(x)


        if self.no_char:
            return order, dst, response, None
        else:
            e_char_2d = e_char_2d.squeeze(1)
            return order, dst, response, e_char_2d, weights
