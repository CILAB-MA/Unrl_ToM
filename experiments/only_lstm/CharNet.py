import torch.nn as nn
import torch as tr


class FC_CharNet(nn.Module):
    def __init__(self,  device, configs):
        super(FC_CharNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.hidden_size = configs['message_output_dim']
        self.attn = nn.MultiheadAttention(configs['message_output_dim'], num_heads=2)
        self.fc_board1 = nn.Linear(configs['num_loc'] * configs['board_feat'], 1024)
        self.fc_board2 = nn.Linear(1024, 256)
        self.fc_board3 = nn.Linear(256, configs['board_output_dim'])
        self.fc_order1 = nn.Linear(configs['order_feat'], 128)
        self.fc_order2 = nn.Linear(128, 32)
        self.fc_order3 = nn.Linear(32, configs['order_output_dim'])
        self.fc_msg1 = nn.Linear(configs['num_msg'] * configs['msg_feat'], 1024)
        self.fc_msg2 = nn.Linear(1024, 128)
        self.fc_msg3 = nn.Linear(128, configs['message_output_dim'])
        self.bn1 = nn.BatchNorm1d(configs['board_output_dim'])
        self.bn2 = nn.BatchNorm1d(configs['order_output_dim'])
        self.bn3 = nn.BatchNorm1d(configs['message_output_dim'])
        self.fc1 = nn.Linear(configs['num_past_step'] * self.hidden_size, 256)
        self.fc2 = nn.Linear(256, configs['char_output_dim'])
        self.device = device

    def forward(self, board, order, message):
        b, num_past, num_step, num_loc, num_board_feature = board.shape
        _, _, _, num_message, num_message_feat = message.shape

        e_char_sum = []
        for p in range(num_past):
            board_past = board[:, p]  # b, num_step, num_loc, num_feature
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

            message_past = message[:, p]  # b, num_step, num_message, num_message_feat
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

            x, _ = self.attn(board_feat, message_feat, order_feat)

            x = x.transpose(1, 0)

            x = x.reshape(b, -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            e_char_sum.append(x)

        e_char_total = tr.stack(e_char_sum)
        final_e_char = sum(e_char_total)

        return final_e_char


class GCN_CharNet(nn.Module):
    def __init__(self, device, configs, norm_adjacency):
        super(GCN_CharNet, self).__init__()
        self.norm_adjacency = norm_adjacency
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool1d(8)

        self.gnn_board1 = GCNLayer(configs['board_feat'], 32)
        self.gnn_board2 = GCNLayer(32, 2)
        self.lstm_board = nn.LSTMCell(162, 162)
        self.fc_board3 = nn.Linear(200 * 162, configs['board_output_dim'])

        self.fc_order1 = nn.Linear(configs['order_feat'], 128)
        self.fc_order2 = nn.Linear(128, 32)
        self.lstm_order = nn.LSTMCell(32, 32)
        self.fc_order3 = nn.Linear(200 * 32, configs['order_output_dim'])

        self.fc_msg1 = nn.Linear(configs['num_msg'] * configs['msg_feat'], 128)
        self.fc_msg2 = nn.Linear(128, 162)
        self.lstm_msg = nn.LSTMCell(162, 162)
        self.fc_msg3 = nn.Linear(200 * 162, configs['message_output_dim'])

        self.board_hidden_size = 162
        self.order_hidden_size = 32
        self.msg_hidden_size = 162

        self.device = device

    def init_hidden(self, batch_size, hidden_size):
        return (tr.zeros(batch_size, hidden_size, device=self.device),
                tr.zeros(batch_size, hidden_size, device=self.device))

    def forward(self, board, order, msg):
        batch_size, num_past, num_step, num_loc, num_obs_feat = board.shape
        _, _, _, num_msg, num_msg_feat = msg.shape
        norm_adjacency = tr.unsqueeze(tr.from_numpy(self.norm_adjacency), 0).repeat(batch_size, 1, 1).to(self.device)

        e_char_board = []
        e_char_order = []
        e_char_msg = []
        for p in range(num_past):
            hx, cx = self.init_hidden(batch_size, self.board_hidden_size)
            board_past = board[:, p]  # b, s, l, f1
            board_feat, _ = self.gnn_board1(board_past, norm_adjacency)
            board_feat, _ = self.gnn_board2(board_feat, norm_adjacency)
            board_feat = board_feat.view(batch_size, num_step, -1)

            board_outs = []
            for step in range(num_step):
                hx, cx = self.lstm_board(board_feat.view(num_step, batch_size, -1)[step], (hx, cx))
                board_outs.append(hx)
            final_board = tr.stack(board_outs, dim=1)
            final_board = final_board.reshape(batch_size, -1)
            final_board = self.fc_board3(final_board)
            e_char_board.append(final_board)

            hx, cx = self.init_hidden(batch_size, self.order_hidden_size)
            order_past = order[:, p]  # b, s, l, f1
            order_feat = self.fc_order1(order_past)
            order_feat = self.fc_order2(order_feat)
            order_feat = order_feat.view(batch_size, num_step, -1)

            order_outs = []
            for step in range(num_step):
                hx, cx = self.lstm_order(order_feat.view(num_step, batch_size, -1)[step], (hx, cx))
                order_outs.append(hx)
            final_order = tr.stack(order_outs, dim=1)
            final_order = final_order.reshape(batch_size, -1)
            final_order = self.fc_order3(final_order)
            e_char_order.append(final_order)

            hx, cx = self.init_hidden(batch_size, self.msg_hidden_size)
            msg_past = msg[:, p]  # b, s, l, f1
            msg_past = msg_past.reshape(batch_size * num_step, -1)
            msg_feat = self.fc_msg1(msg_past)
            msg_feat = self.fc_msg2(msg_feat)
            msg_feat = msg_feat.view(batch_size, num_step, -1)

            msg_outs = []
            for step in range(num_step):
                hx, cx = self.lstm_msg(msg_feat.view(num_step, batch_size, -1)[step], (hx, cx))
                msg_outs.append(hx)
            final_msg = tr.stack(msg_outs, dim=1)
            final_msg = final_msg.reshape(batch_size, -1)
            final_msg = self.fc_msg3(final_msg)
            e_char_msg.append(final_msg)

        e_char_board_total = tr.stack(e_char_board)
        e_char_order_total = tr.stack(e_char_order)
        e_char_msg_total = tr.stack(e_char_msg)

        final_e_char_board = sum(e_char_board_total)
        final_e_char_order = sum(e_char_order_total)
        final_e_char_msg = sum(e_char_msg_total)

        final_e_char = final_e_char_board + final_e_char_order + final_e_char_msg
        return final_e_char


class Basic_GCN_CharNet(nn.Module):
    def __init__(self, device, configs, norm_adjacency):
        super(Basic_GCN_CharNet, self).__init__()
        self.hidden_size = 324 + configs['message_output_dim'] + configs['order_output_dim']
        self.norm_adjacency = norm_adjacency
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool1d(8)

        self.gnn_board1 = GCNLayer(configs['board_feat'], 32)
        self.gnn_board2 = GCNLayer(32, 8)
        self.gnn_board3 = GCNLayer(8, configs['board_output_dim'])

        self.fc_order1 = nn.Linear(configs['order_feat'], 128)
        self.fc_order2 = nn.Linear(128, 32)
        self.fc_order3 = nn.Linear(32, configs['order_output_dim'])

        self.fc_msg1 = nn.Linear(configs['num_msg'] * configs['msg_feat'], 1024)
        self.fc_msg2 = nn.Linear(1024, 128)
        self.fc_msg3 = nn.Linear(128, configs['message_output_dim'])

        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(configs['num_past_step'] * self.hidden_size, configs['char_output_dim'])

        self.device = device

    def init_hidden(self, batch_size, hidden_size):
        return (tr.zeros(batch_size, hidden_size, device=self.device),
                tr.zeros(batch_size, hidden_size, device=self.device))

    def forward(self, board, order, msg):
        batch_size, num_past, num_step, num_loc, num_obs_feat = board.shape
        _, _, _, num_msg, num_msg_feat = msg.shape
        norm_adjacency = tr.unsqueeze(tr.from_numpy(self.norm_adjacency), 0).repeat(batch_size, 1, 1).to(self.device)

        e_char_sum = []
        for p in range(num_past):
            hx, cx = self.init_hidden(batch_size, self.hidden_size)
            board_past = board[:, p]  # b, s, l, f1
            board_feat, _ = self.gnn_board1(board_past, norm_adjacency)
            board_feat, _ = self.gnn_board2(board_feat, norm_adjacency)
            board_feat, _ = self.gnn_board3(board_feat, norm_adjacency)
            board_feat = board_feat.view(batch_size, num_step, -1)

            order_past = order[:, p]  # b, num_step, order_feature
            order_past = order_past.reshape(batch_size * num_step, -1)
            order_feat = self.fc_order1(order_past)
            order_feat = self.fc_order2(order_feat)
            order_feat = self.fc_order3(order_feat)
            order_feat = order_feat.view(batch_size, num_step, -1)

            message_past = msg[:, p]  # b, num_step, num_message, num_message_feat
            message_past = message_past.reshape(batch_size * num_step, -1)
            message_feat = self.fc_msg1(message_past)
            message_feat = self.fc_msg2(message_feat)
            message_feat = self.fc_msg3(message_feat)
            message_feat = message_feat.view(batch_size, num_step, -1)

            x_feat = tr.cat([board_feat, order_feat, message_feat], dim=-1)
            x_feat = x_feat.transpose(1, 2)  # batch, step, feature
            x_feat = self.bn(x_feat)  # batch, feature, step
            x_feat = x_feat.permute(2, 0, 1)  # step, batch, feature
            outs = []
            for step in range(num_step):
                hx, cx = self.lstm(x_feat.view(num_step, batch_size, -1)[step], (hx, cx))
                outs.append(hx)
            x = tr.stack(outs, dim=0)
            x = x.transpose(1, 0)
            x = x.reshape(batch_size, -1)
            x = self.fc(x)
            e_char_sum.append(x)

        e_char_total = tr.stack(e_char_sum)
        final_e_char = sum(e_char_total)

        return final_e_char


class GCNLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()

        self.in_features = input_dim
        self.out_features = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = nn.ReLU()

    def forward(self, x, norm_adjacency):
        batch, num_step, num_loc, num_feature = x.shape

        out_list = list()
        for step in range(num_step):
            x_step = x[:, step]
            out = self.linear(x_step)
            out = tr.matmul(norm_adjacency, out)
            out = self.activation(out)
            out_list.append(out)

        final_output = tr.stack(out_list, dim=1)
        return final_output, norm_adjacency
