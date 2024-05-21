import torch.nn as nn
import torch as tr


class FC_CharNet(nn.Module):
    def __init__(self,  device, configs):
        super(FC_CharNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.hidden_size = configs['message_output_dim']# + configs['board_output_dim'] + configs['order_output_dim']
        self.attn1 = nn.MultiheadAttention(self.hidden_size, num_heads=2)
        #self.attn2 = nn.MultiheadAttention(self.hidden_size, num_heads=2)
        self.fc_board1 = nn.Linear(configs['num_board'] * configs['board_feat'], 1024)
        self.fc_board2 = nn.Linear(1024, 256)
        self.fc_board3 = nn.Linear(256, configs['board_output_dim'])
        self.fc_order1 = nn.Linear(configs['order_feat'], 128)
        self.fc_order2 = nn.Linear(128, 32)
        self.fc_order3 = nn.Linear(32, configs['order_output_dim'])
        self.fc_msg1 = nn.Linear(2 * configs['msg_feat'], 64)
        self.fc_msg2 = nn.Linear(64, 32)
        self.fc_msg3 = nn.Linear(32, configs['message_output_dim'])
        self.bn1 = nn.BatchNorm1d(configs['board_output_dim'])
        self.bn2 = nn.BatchNorm1d(configs['order_output_dim'])
        self.bn3 = nn.BatchNorm1d(configs['message_output_dim'])
        self.fc1 = nn.Linear(configs['num_past_step'] * self.hidden_size, 256)
        self.fc2 = nn.Linear(256 + 14, configs['char_output_dim'])
        self.device = device

    def forward(self, board, order, message, other_ind, me_ind):
        b, num_past, num_step, num_loc, num_board_feature = board.shape
        _, _, _, num_message_feat = message.shape

        e_char_sum = []
        for p in range(num_past):
            board_past = board[:, p] # b, num_step, num_loc, num_feature
            board_past = board_past.reshape(b * num_step, -1)
            board_feat = self.relu(self.fc_board1(board_past))
            board_feat = self.relu(self.fc_board2(board_feat))
            board_feat = self.fc_board3(board_feat)
            board_feat = board_feat.view(b, num_step, -1)
            board_feat = board_feat.transpose(1, 2)
            board_feat = self.bn1(board_feat)
            board_feat = self.relu(board_feat)

            order_past = order[:, p]  # b, num_step, order_feature
            order_past = order_past.reshape(b * num_step, -1)
            order_feat = self.relu(self.fc_order1(order_past))
            order_feat = self.relu(self.fc_order2(order_feat))
            order_feat = self.fc_order3(order_feat)
            order_feat = order_feat.view(b, num_step, -1)
            order_feat = order_feat.transpose(1, 2)
            order_feat = self.bn2(order_feat)
            order_feat = self.relu(order_feat)

            message_past = message[:, p]  # b, num_step, num_message_feat
            message_past = message_past.reshape(b * num_step, -1)
            message_feat = self.relu(self.fc_msg1(message_past))
            message_feat = self.relu(self.fc_msg2(message_feat))
            message_feat = self.fc_msg3(message_feat)
            message_feat = message_feat.view(b, num_step, -1)
            message_feat = message_feat.transpose(1, 2)
            message_feat = self.bn3(message_feat)
            message_feat = self.relu(message_feat)

            board_feat = board_feat.permute(2, 0, 1)
            order_feat = order_feat.permute(2, 0, 1)
            message_feat = message_feat.permute(2, 0, 1)
            x, weights = self.attn1(board_feat, message_feat, order_feat)
            #x2, weights2 = self.attn2(order_feat, board_feat, message_feat)
            #x = (x1 + x2) / 2
            #weights = (weights1 + weights2) / 2
            x = x.transpose(1, 0)
            x = x.reshape(b, -1)
            x = self.relu(self.fc1(x))
            x = tr.cat([x, other_ind[:, p], me_ind[:, p]], dim=-1)
            x = self.fc2(x)
            e_char_sum.append(x)

        e_char_total = tr.stack(e_char_sum)
        final_e_char = sum(e_char_total)

        return final_e_char, weights
