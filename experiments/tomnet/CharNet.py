import torch.nn.functional as F
import torch.nn as nn
import torch as tr
import math
import numpy as np

class FC_CharNet(nn.Module):
    def __init__(self,  device, configs):
        super(FC_CharNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.hidden_size = configs['message_output_dim'] + configs['board_output_dim'] + configs['order_output_dim']
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.fc_board1 = nn.Linear(configs['num_loc'] * configs['board_feat'], 1024)
        self.fc_board2 = nn.Linear(1024, 256)
        self.fc_board3 = nn.Linear(256, configs['board_output_dim'])
        self.fc_order1 = nn.Linear(configs['order_feat'], 128)
        self.fc_order2 = nn.Linear(128, configs['order_output_dim'])
        self.fc_msg1 = nn.Linear(2 * configs['msg_feat'], 64)
        self.fc_msg2 = nn.Linear(64, configs['message_output_dim'])
        self.fc1 = nn.Linear(self.hidden_size + 2 * configs['num_agent'], configs['char_output_dim'])
        self.fc2 = nn.Linear(configs['char_output_dim'], configs['char_output_dim'])
        self.device = device
        self.configs = configs

    def init_hidden(self, batch_size):
        return (tr.randn(batch_size, self.hidden_size, device=self.device),
                tr.randn(batch_size, self.hidden_size, device=self.device))

    def forward(self, board, order, message, other_ind, me_ind):
        b, num_past, num_step, num_loc, num_board_feature = board.shape
        _, _, _, num_message_feat = message.shape

        e_char_sum = []
        for p in range(num_past):
            hx, cx = self.init_hidden(b)
            board_past = board[:, p]  # b, num_step, num_loc, num_feature
            board_past = board_past.reshape(b * num_step, -1)
            board_feat = self.relu(self.fc_board1(board_past))
            board_feat = self.relu(self.fc_board2(board_feat))
            board_feat = self.fc_board3(board_feat)
            board_feat = board_feat.view(b, num_step, -1)
            board_feat = board_feat.transpose(1, 2)
            # board_feat = self.bn1(board_feat)
            board_feat = self.relu(board_feat)

            order_past = order[:, p]  # b, num_step, order_feature
            order_past = order_past.reshape(b * num_step, -1)
            order_feat = self.relu(self.fc_order1(order_past))
            order_feat = self.fc_order2(order_feat)
            order_feat = order_feat.view(b, num_step, -1)
            order_feat = order_feat.transpose(1, 2)
            # order_feat = self.bn2(order_feat)
            order_feat = self.relu(order_feat)

            message_past = message[:, p]  # b, num_step, num_message_feat
            message_past = message_past.reshape(b * num_step, -1)
            message_feat = self.relu(self.fc_msg1(message_past))
            message_feat = self.fc_msg2(message_feat)
            message_feat = message_feat.view(b, num_step, -1)
            message_feat = message_feat.transpose(1, 2)
            # message_feat = self.bn3(message_feat)
            message_feat = self.relu(message_feat)

            x_feat = tr.cat([board_feat, order_feat, message_feat], dim=1)  # batch, feature, step
            # x_feat = self.bn(x_feat) #batch, feature, step
            x_feat_out = x_feat.permute(2, 0, 1)  # step, batch, feature

            # mental_net
            outs = []
            for step in range(num_step):
                hx, cx = self.lstm(x_feat_out[step], (hx, cx))
                outs.append(hx)
            x = tr.stack(outs, dim=0)
            x = x.transpose(1, 0)
            x = x.sum(dim=1)
            q = x.reshape(b, -1)

            # embed query state
            # query = self.relu(self.query_fc1(x))
            # q = self.relu(self.query_fc2(query))
            x = tr.cat([q, other_ind[:, p], me_ind[:, p]], dim=-1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            e_char_sum.append(x)

        e_char_total = tr.stack(e_char_sum)
        final_e_char = sum(e_char_total)
        # final_e_char = e_char_total.mean(dim=0)

        return final_e_char, e_char_total

class GCN_CharNet(nn.Module):
    def __init__(self,  A, device, configs):
        super(GCN_CharNet, self).__init__()
        num_blocks = 1
        inter_emb_size = 32  # 120
        learnable_A = False
        dropout = 0.4  # 0.2 or 0.4
        residual_linear = False
        use_global_pooling = False
        layerdrop = 0

        # board state blocks
        self.board_blocks = nn.ModuleList()
        self.board_blocks.append(
            DiplomacyModelBlock(in_size=configs['board_feat'], out_size=inter_emb_size, A=A,
                                residual=False, learnable_A=learnable_A, dropout=dropout,
                                residual_linear=residual_linear, use_global_pooling=use_global_pooling))

        for _ in range(num_blocks - 1):
            self.board_blocks.append(
                DiplomacyModelBlock(in_size=inter_emb_size, out_size=inter_emb_size, A=A,
                                    residual=True, learnable_A=learnable_A, dropout=dropout,
                                    residual_linear=residual_linear, use_global_pooling=use_global_pooling))

        if layerdrop > 1e-5:
            assert 0 < layerdrop <= 1.0, layerdrop
            self.layerdrop_rng = np.random.RandomState(0)
        else:
            self.layerdrop_rng = None
        self.layerdrop = layerdrop

        self.relu = nn.ReLU(inplace=True)
        self.hidden_size = configs['message_output_dim'] + configs['board_output_dim'] + configs['order_output_dim']
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.fc_board1 = nn.Linear(configs['num_loc'] * inter_emb_size, 1024)
        self.fc_board2 = nn.Linear(1024, 256)
        self.fc_board3 = nn.Linear(256, configs['board_output_dim'])
        self.fc_order1 = nn.Linear(configs['order_feat'], 128)
        self.fc_order2 = nn.Linear(128, configs['order_output_dim'])
        self.fc_msg1 = nn.Linear(2 * configs['msg_feat'], 64)
        self.fc_msg2 = nn.Linear(64, configs['message_output_dim'])
        self.fc1 = nn.Linear(self.hidden_size + 2 * configs['num_agent'], configs['char_output_dim'])
        self.fc2 = nn.Linear(configs['char_output_dim'], configs['char_output_dim'])
        self.device = device
        self.configs = configs

    def init_hidden(self, batch_size):
        return (tr.randn(batch_size, self.hidden_size, device=self.device),
                tr.randn(batch_size, self.hidden_size, device=self.device))

    def forward(self, board, order, message, other_ind, me_ind):
        b, num_past, num_step, num_loc, num_board_feature = board.shape
        _, _, _, num_message_feat = message.shape

        def apply_blocks_with_layerdrop(blocks, tensor):
            for i, block in enumerate(blocks):
                drop = (
                        i > 0
                        and self.training
                        and self.layerdrop_rng is not None
                        and self.layerdrop_rng.uniform() < self.layerdrop
                )
                if drop:
                    # To make distrubited happy we need to have grads for all params.
                    dummy = sum(w.sum() * 0 for w in block.parameters())
                    tensor = dummy + tensor
                else:
                    tensor = block(tensor)
            return tensor

        e_char_sum = []
        for p in range(num_past):
            hx, cx = self.init_hidden(b)
            board_feat = apply_blocks_with_layerdrop(self.board_blocks, board[:, p])
            board_feat = board_feat.view(b * num_step, -1)
            board_feat = self.relu(self.fc_board1(board_feat))
            board_feat = self.relu(self.fc_board2(board_feat))
            board_feat = self.fc_board3(board_feat)
            board_feat = board_feat.view(b, num_step, -1)
            board_feat = board_feat.transpose(1, 2)
            board_feat = self.relu(board_feat)

            order_past = order[:, p]  # b, num_step, order_feature
            order_past = order_past.reshape(b * num_step, -1)
            order_feat = self.relu(self.fc_order1(order_past))
            order_feat = self.fc_order2(order_feat)
            order_feat = order_feat.view(b, num_step, -1)
            order_feat = order_feat.transpose(1, 2)
            order_feat = self.relu(order_feat)

            message_past = message[:, p]  # b, num_step, num_message_feat
            message_past = message_past.reshape(b * num_step, -1)
            message_feat = self.relu(self.fc_msg1(message_past))
            message_feat = self.fc_msg2(message_feat)
            message_feat = message_feat.view(b, num_step, -1)
            message_feat = message_feat.transpose(1, 2)
            message_feat = self.relu(message_feat)

            x_feat = tr.cat([board_feat, order_feat, message_feat], dim=1)  # batch, feature, step
            x_feat_out = x_feat.permute(2, 0, 1)  # step, batch, feature

            outs = []
            for step in range(num_step):
                hx, cx = self.lstm(x_feat_out[step], (hx, cx))
                outs.append(hx)
            x = tr.stack(outs, dim=0)
            x = x.transpose(1, 0)
            x = x.sum(dim=1)
            q = x.reshape(b, -1)

            # embed query state
            x = tr.cat([q, other_ind[:, p], me_ind[:, p]], dim=-1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            e_char_sum.append(x)

        e_char_total = tr.stack(e_char_sum)
        final_e_char = sum(e_char_total)

        return final_e_char, e_char_total


class DiplomacyModelBlock(nn.Module):
    def __init__(self, *, in_size, out_size, A, dropout, residual=True, learnable_A=False,
                 residual_linear=False, use_global_pooling=False):
        super().__init__()
        self.graph_conv = GraphConv(in_size, out_size, A, learnable_A=learnable_A)
        self.batch_norm = nn.BatchNorm1d(A.shape[0])
        self.dropout = nn.Dropout(dropout or 0.0)
        self.residual = residual
        self.residual_linear = residual_linear
        if residual_linear:
            self.residual_lin = nn.Linear(in_size, out_size)
        self.use_global_pooling = use_global_pooling
        if use_global_pooling:
            self.post_pool_lin = nn.Linear(out_size, out_size, bias=False)

    def forward(self, x):
        # Shape [batch_idx, location, channel]
        batch, num_step, num_loc, num_feature = x.shape

        y_list = list()
        for step in range(num_step):
            y = self.graph_conv(x[:, step])
            if self.residual_linear:
                y += self.residual_lin(x[:, step])
            y = self.batch_norm(y)
            if self.use_global_pooling:
                # Global average pool over location
                g = tr.mean(y, dim=1, keepdim=True)
                g = self.dropout(g)
                g = self.post_pool_lin(g)
                # Add back transformed-pooled values as per-channel biases
                y += g
            y = F.relu(y)
            y = self.dropout(y)
            if self.residual:
                y += x[:, step]
            y_list.append(y)

        y_list_final = tr.stack(y_list, dim=1)

        return y_list_final


class GraphConv(nn.Module):
    def __init__(self, in_size, out_size, A, learnable_A=False):
        super().__init__()
        """
        A -> (81, 81)
        """
        self.A = nn.Parameter(A).requires_grad_(learnable_A)
        self.W = nn.Parameter(he_init((len(self.A), in_size, out_size)))
        self.b = nn.Parameter(tr.zeros(1, 1, out_size))

    def forward(self, x):
        """Computes A*x*W + b

        x -> (B, 81, in_size)
        returns (B, 81, out_size)
        """

        x = x.transpose(0, 1)  # (b, N, in )               => (N, b, in )
        x = tr.matmul(x, self.W)  # (N, b, in) * (N, in, out) => (N, b, out)
        x = x.transpose(0, 1)  # (N, b, out)               => (b, N, out)
        x = tr.matmul(self.A, x)  # (b, N, N) * (b, N, out)   => (b, N, out)
        x += self.b

        return x


def he_init(shape):
    fan_in = shape[-2] if len(shape) >= 2 else shape[-1]
    init_range = math.sqrt(2.0 / fan_in)
    return tr.randn(shape) * init_range
