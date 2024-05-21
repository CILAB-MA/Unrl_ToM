import numpy as np
import copy
from environment.l1_negotiation import Negotiation
from environment.order import Order

class MessageMaker:

    def __init__(self, num_agent, powers, static_infos):

        self.num_agent = num_agent
        self.powers = powers

        self.parser = Negotiation()
        self.order_parser = Order()
        self.locs = static_infos['locs']
        self.units = ['A', 'F']
        self.orders = ['H', '-', 'S', 'C', 'VIA']

        self.nego_level = ['P', 'A', 'R']  # propose(send), accept(recv), reject(recv), withdraw
        self.nego_lang = ['PROPOSE', 'ACCEPT', 'REJECT']
        self.deal_level = ['C', 'A'] # Commit, Agree
        self.deal_lang = ['COMMIT', 'AGREE']
        self.content_level = ['P', 'AL', 'D']  # predicate, do, and, not
        self.content_lang = ['PEACE', 'ALLIANCE', 'DO']
        self.ally = ['ALLY', 'ENEMY']
        self.max_len = 20 # length of ally negotiation(15)
        self.dict = self.locs + self.orders + self.units + self.nego_lang + self.ally + self.deal_lang + self.content_lang + self.powers

    def _split_conts(self, conts):
        if conts[0] == 'PEACE':
            conts = [conts[0]] + conts[1].split(',')
        elif conts[0] == 'ALLIANCE':
            conts = [conts[0]] + ['ALLY'] + conts[1].split(',') + ['ENEMY'] + conts[2].split(',')
        elif conts[0] == 'DO':
            conts = [conts[0]] + conts[1].split()
        return conts

    def make_nlp_message(self, messages, me, other):

        sorted_messages = []

        all_msg = np.zeros((40, 40), dtype=np.int8)
        masking = np.full((40,), False)
        for k, vs in messages.items():
            sorted_messages += vs

        # private message
        sorted_messages = sorted(sorted_messages, key=lambda x : x[1])

        messages_send = sorted_messages[0::2]
        messages_recv = sorted_messages[1::2]
        cnt = -1
        me_send = []
        for send, recv in zip(messages_send, messages_recv):
            send_msg, _ = send
            recv_msg, _ = recv
            send_msg = send_msg['message']
            recv_msg = recv_msg['message']
            # parsing message
            sr1, nego_l1, deals1, conts1, sender1, receiver1, _ = self.parser.parse(send_msg)
            sr2, nego_l2, deals2, conts2, receiver2, sender2, _ = self.parser.parse(recv_msg)

            # error check
            if (sr1 != 'S') or (sr2 != 'R') or (sender1 != sender2) or (receiver1 != receiver2) or (conts1 != conts2):
                assert 'Something Message format is wrong or Sorting is wrong! Check.'
            sender_ind = self.powers.index(sender1)
            receiver_ind = self.powers.index(receiver1)
            if (sender_ind not in [me, other]) or (receiver_ind not in [me, other]):
                continue
            cnt += 1
            words1 = [nego_l1, sender1, receiver1, deals1[0]] + self._split_conts(conts1)
            words1_num = [self.dict.index(w1) for w1 in words1]
            words2 = [nego_l2, sender2, receiver2, deals2[0]] + self._split_conts(conts2)
            words2_num = [self.dict.index(w2) for w2 in words2]
            send_msg_state = np.full(self.max_len, -1, dtype=np.int8)
            recv_msg_state = np.full(self.max_len, -1, dtype=np.int8)
            send_msg_state[:len(words1_num)] = words1_num
            recv_msg_state[:len(words2_num)] = words2_num

            msg_state = np.concatenate([send_msg_state, recv_msg_state])
            masking[cnt] = True
            if sender1 == me:
                me_send.append(cnt)
            all_msg[cnt] = msg_state

        return all_msg

    def make_message(self, messages, me, other, loc2power):

        sorted_messages = []

        all_msg = np.zeros((40, self.num_agent * 5 + 8 + 2), dtype=np.int8)
        for k, vs in messages.items():
            sorted_messages += vs

        # private message
        sorted_messages = sorted(sorted_messages, key=lambda x : x[1])

        messages_send = sorted_messages[0::2]
        messages_recv = sorted_messages[1::2]
        cnt = -1
        me_send = np.zeros(40, dtype=np.int8)
        for send, recv in zip(messages_send, messages_recv):
            send_msg, _ = send
            recv_msg, _ = recv
            send_msg = send_msg['message']
            recv_msg = recv_msg['message']
            # parsing message
            sr1, nego_l1, deals1, conts1, sender1, receiver1, _ = self.parser.parse(send_msg)
            sr2, nego_l2, deals2, conts2, receiver2, sender2, _ = self.parser.parse(recv_msg)

            # error check
            if (sr1 != 'S') or (sr2 != 'R') or (sender1 != sender2) or (receiver1 != receiver2) or (conts1 != conts2):
                assert 'Something Message format is wrong or Sorting is wrong! Check.'
            sender_ind = self.powers.index(sender1)
            receiver_ind = self.powers.index(receiver1)
            nego_ind1 = self.nego_lang.index(nego_l1)
            nego_ind2 = self.nego_lang.index(nego_l2)

            if (sender_ind not in [me, other]) or (receiver_ind not in [me, other]):
                continue
            send_msg_state = np.zeros(self.num_agent * 5 + 8, dtype=np.int8)
            recv_msg_state = np.zeros(2, dtype=np.int8)

            # put in the sender, receiver
            send_msg_state[sender_ind] = 1
            send_msg_state[receiver_ind + 7] = 1

            cnt += 1

            # put the nego
            send_msg_state[nego_ind1 + 14] = 1
            recv_msg_state[nego_ind2 - 1] = 1

            # put the peace
            if conts1[0] == 'PEACE':
                powers = conts1[1].split(',')
                for p in powers:
                    peace_ind = self.powers.index(p)
                    send_msg_state[peace_ind + 17] = 1

            # put the peace, enemy
            elif conts1[0] == 'ALLIANCE':
                allies = conts1[1].split(',')
                enemies = conts1[2].split(',')
                for a in allies:
                    ally_ind = self.powers.index(a)
                    send_msg_state[ally_ind + 17] = 1

                for e in enemies:
                    enemy_ind = self.powers.index(e)
                    send_msg_state[enemy_ind + 24] = 1

            # put the order, target
            elif conts1[0] == 'DO':
                _, _, order_type, _, dst_infos = self.order_parser.parse(conts1[1], loc2power)
                order_ind = self.orders.index(order_type)
                if dst_infos[1] != None:
                    dst_ind = self.powers.index(dst_infos[1])
                else:
                    dst_ind = 7
                send_msg_state[order_ind + 31] = 1
                send_msg_state[dst_ind + 35] = 1

            if sender_ind == me:
                me_send[cnt] = 1
            msg_state = np.concatenate([send_msg_state, recv_msg_state])
            all_msg[cnt] = msg_state

        return all_msg, me_send.sum(), me_send

    def inverse_nlp_message(self, message):
        message_word = [self.dicts[m] for m in message if not m == -1]
        return message_word











