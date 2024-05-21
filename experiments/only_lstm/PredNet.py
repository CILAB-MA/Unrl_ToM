from .CharNet import *
import torch.nn.functional as F

class Only_lstm_PredNet(nn.Module):
    def __init__(self, device, norm_adjacency, configs):
        super(Only_lstm_PredNet, self).__init__()
        self.hidden_size = 2 * configs['msg_feat'] + (configs['num_board'] * configs['board_feat']) + configs['order_feat']
        self.norm_adjacency = norm_adjacency
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear((self.hidden_size + configs['src_dim'] + configs['send_dim'] + (configs['num_board'] * configs['board_feat']) + 14), 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.device = device
        self.no_char = configs['no_char']
        self.softmax = nn.Softmax(dim=-1)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.order_head = nn.Sequential(nn.ReLU(),
                                        nn.Linear(256, 4),
                                        nn.LogSoftmax(dim=-1))

        self.dst_head = nn.Sequential(nn.ReLU(),
                                      nn.Linear(256, 81),
                                      nn.LogSoftmax(dim=-1))
        self.response_head = nn.Sequential(nn.ReLU(),
                                      nn.Linear(256, 2),
                                      nn.LogSoftmax(dim=-1))

    def init_hidden(self, batch_size):
        return (tr.randn(batch_size, self.hidden_size, device=self.device),
                tr.randn(batch_size, self.hidden_size, device=self.device))

    def forward(self, curr_board, curr_src, curr_msg, curr_send, curr_order, me_ind, other_ind):

        curr_board_last = curr_board[:, -1, :]
        prep_curr_order = curr_order[:, :, :]
        prep_curr_board = curr_board[:, :  -1, : ]
        prep_curr_msg = curr_msg[:, :  -1, :]

        _, curr_num_order, _ = prep_curr_order.shape
        batch, curr_num_step, num_loc, num_board_feature = prep_curr_board.shape
        _, curr_num_msg, _ = prep_curr_msg.shape

        hx, cx = self.init_hidden(batch)

        order_feat = prep_curr_order.reshape(batch, curr_num_order, -1)
        obs_feature = prep_curr_board.reshape(batch, curr_num_step, -1)
        message_feat = prep_curr_msg.reshape(batch, curr_num_msg, -1)

        x_concat = tr.cat([order_feat, obs_feature, message_feat], dim=-1)
        x_concat = x_concat.reshape(batch, -1)

        outs = []
        for step in range(curr_num_step):
            hx, cx = self.lstm(x_concat.reshape(curr_num_step, batch, -1)[step], (hx, cx))
            outs.append(hx)
        final_out = tr.stack(outs, dim=1)
        final_out_last = final_out[:, -1:, :]
        curr_board_last = curr_board_last.reshape(batch, 1, -1)

        x = tr.cat([final_out_last, curr_src, curr_send, curr_board_last, me_ind, other_ind], dim=-1)
        x = x.reshape(batch, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        order = self.order_head(x)
        dst = self.dst_head(x)
        response = self.response_head(x)

        return order, dst, response, None