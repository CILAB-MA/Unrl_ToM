import numpy as np
import copy
from environment.l1_negotiation import Negotiation

class Np_state:

    def __init__(self, num_agent, static_infos):
        self.num_agent = num_agent
        self.negos = ['PROPOSE', 'ACCEPT', 'REJECT']
        self.contents = ['PEACE', 'ALLIANCE', 'DO']
        self.parser = Negotiation()
        self.static_infos = static_infos

        self.locs = static_infos['locs']
        self.powers = list(static_infos['powers'])

        self.one_msg_len = 5 * self.num_agent + len(self.negos)
        self.msg_shape = (4, 2 * self.one_msg_len)


        self.locs = self.static_infos['locs']

        self.units = ['A', 'F', None]
        self.orders = ['H', '-', 'S', 'C']
        self.num_order = 4

        self.num_power = len(self.powers)
        self.areas = ['LAND', 'WATER', 'PORT', 'COAST']
        self.area_types = static_infos['area_type']

    def make_time_step(self, game_obj, last_step=None):
        curr_phase = game_obj.get_current_phase()
        first_phase = game_obj.map.first_year
        if curr_phase == 'COMPLETED':
            return last_step + 1
        return int(curr_phase[1:5]) - first_phase


    def check_done(self, game_obj):
        return game_obj.is_game_done

    def message_state_maker(self, messages, me, other):
        '''
        msg shape
        sender num_agent
        receiver num_agent
        nego 3
        peace num_agent
        alliance num_agent
        against num_agent
        '''

        all_msg = np.zeros(self.msg_shape)
        power_cnts = [0] * self.num_agent
        sender_response = np.full((4,), 2) # 0 Accept 1 Reject 2 Propose
        sr_padding = np.full((4,), False) # send -> recv or recv -> send
        masking = np.full((4,), False)

        sorted_messages = []
        for k, v in messages.items():
            if k == "message":
                sorted_messages.append(v)

        # private message
        sorted_messages = sorted(sorted_messages, key=lambda x : x[1])

        messages_send = sorted_messages[0::2]
        messages_recv = sorted_messages[1::2]
        for send, recv in zip(messages_send, messages_recv):
            send_msg, _ = send
            recv_msg, _ = recv
            # parsing message
            sr1, nego_l1, deals1, conts1, sender1, receiver1, _ = self.parser.parse(send_msg)
            sr2, nego_l2, deals2, conts2, receiver2, sender2, _ = self.parser.parse(recv_msg)
            send_msg_state = np.zeros(self.one_msg_len)
            recv_msg_state = np.zeros(self.one_msg_len)
            # error check
            if (sr1 != 'S') or (sr2 != 'R') or (sender1 != sender2) or (receiver1 != receiver2) or (conts1 != conts2):
                assert 'Something Message format is wrong or Sorting is wrong! Check.'
            sender_ind = self.powers.index(sender1)
            receiver_ind = self.powers.index(receiver1)
            if (sender_ind not in [me, other]) or (receiver_ind not in [me, other]):
                continue

            send_msg_state[sender_ind] = 1
            send_msg_state[self.num_agent+receiver_ind] = 1
            recv_msg_state[receiver_ind] = 1
            recv_msg_state[self.num_agent+sender_ind] = 1
            nego1_ind = self.negos.index(nego_l1)
            nego2_ind = self.negos.index(nego_l2)
            send_msg_state[2 * self.num_agent + nego1_ind] = 1
            recv_msg_state[2 * self.num_agent + nego2_ind] = 1

            cont_ind = conts1.index(conts1[0])

            # for peace, alliance
            if conts1[0] == 'PEACE':
                peaces = conts1[1].split(',')
                for peace in peaces:
                    peace_ind = self.powers.index(peace)
                    send_msg_state[2 * self.num_agent + len(self.negos) + peace_ind] = 1
                    recv_msg_state[2 * self.num_agent + len(self.negos) + peace_ind] = 1

            elif conts1[0] == 'ALLIANCE':
                allies = conts1[1].split(',')
                enemies = conts1[2].split(',')
                for ally in allies:
                    ally_ind = self.powers.index(ally)
                    send_msg_state[3 * self.num_agent + len(self.negos) + ally_ind] = 1
                    recv_msg_state[3 * self.num_agent + len(self.negos) + ally_ind] = 1
                for enemy in enemies:
                    enemy_ind = self.powers.index(enemy)
                    send_msg_state[4 * self.num_agent + len(self.negos) + enemy_ind] = 1
                    recv_msg_state[4 * self.num_agent + len(self.negos) + enemy_ind] = 1
            else:
                continue
            msg_state = np.concatenate([send_msg_state, recv_msg_state])
            all_msg[power_cnts[sender_ind]] = msg_state
            sr_padding[power_cnts[sender_ind]] = True
            masking[power_cnts[sender_ind]] = True
            if nego_l2 == 'ACCEPT':
                sender_response[power_cnts[sender_ind]] = 0
            else:
                sender_response[power_cnts[sender_ind]] = 1
            power_cnts[sender_ind] = power_cnts[sender_ind] + 1
            power_cnts[receiver_ind] = power_cnts[receiver_ind] + 1


        return all_msg, sr_padding, masking, sender_response


    def prev_order_state_maker(self, obss):

        scs = copy.deepcopy(self.static_infos['scs'])
        # init prev order state
        loc_units = np.zeros((len(self.locs), len(self.units)))  # Unit (A, F, None)
        loc_powers = np.zeros((len(self.locs), self.num_agent + 1))  # Owner (7 + None) -> will change
        loc_orders = np.zeros((len(self.locs), self.num_order + 1))  # Order + None
        loc_src_powers = np.zeros((len(self.locs), self.num_agent + 1))
        loc_dst_powers = np.zeros((len(self.locs), self.num_agent + 1))
        loc_sc_powers = np.zeros((len(self.locs), self.num_agent + 1))

        # record the owner of supply centers
        owner = dict()
        state = obss['state']
        for power_name in state['units']:
            for unit in state['units'][power_name]:
                loc = unit.split()[-1]
                owner[loc] = power_name

        for power_name in state['centers']:
            if power_name == 'UNOWNED':
                continue
            for sc in state['centers'][power_name]:
                for loc in [map_loc for map_loc in self.locs if map_loc == sc]:
                    if loc not in owner:
                        owner[loc] = power_name
                    power_ind = self.powers.index(power_name)
                    sc_ind = self.locs.index(sc)
                    loc_sc_powers[sc_ind, power_ind] = 1
                scs.remove(sc)

        # remained
        for sc in scs:
            for loc in [map_loc for map_loc in self.locs if map_loc == sc]:
                sc_ind = self.locs.index(loc)
                loc_sc_powers[sc_ind, -1] = 1

        # parsing orders
        for power_name in obss['orders']:
            for order in obss['orders'][power_name]:
                word = order.split()

                if (len(word) <= 2) or word[2] not in self.orders:
                    print('Unsupported order')
                    continue

                unit_type, unit_loc, order_type = word[:3]
                unit_ind = self.units.index(unit_type)
                order_ind = self.orders.index(order_type)
                unit_loc_ind = self.locs.index(unit_loc)

                # for hold order
                if order_type == 'H':
                    loc_src_powers[unit_loc_ind, -1] = 1
                    loc_dst_powers[unit_loc_ind, -1] = 1

                # for mover order
                elif order_type == '-':
                    dst = word[-1]
                    if dst not in owner:
                        dst_power_ind = -1
                    else:
                        dst_power_ind = self.powers.index(owner[dst])
                    loc_src_powers[unit_loc_ind, -1] = 1
                    loc_dst_powers[unit_loc_ind, dst_power_ind] = 1

                # for support hold
                elif order_type == 'S' and '-' not in word:
                    src = word[-1]
                    if src not in owner:
                        src_power_ind = -1
                    else:
                        src_power_ind = self.powers.index(owner[src])
                    loc_src_powers[unit_loc_ind, src_power_ind] = 1
                    loc_dst_powers[unit_loc_ind, -1] = 1

                # for support move and convoy
                elif (order_type in ['S', 'C']) and ('-' in word):
                    src = word[word.index('-') - 1]
                    dst = word[-1]
                    if src not in owner:
                        src_power_ind = -1
                    else:
                        src_power_ind = self.powers.index(owner[src])
                    if dst not in owner:
                        dst_power_ind = -1
                    else:
                        dst_power_ind = self.powers.index(owner[dst])
                    loc_src_powers[unit_loc_ind, src_power_ind] = 1
                    loc_dst_powers[unit_loc_ind, dst_power_ind] = 1

                else:
                    print('Wrong Order!')
        loc_units[(np.sum(loc_units, axis=1) == 0, -1)] = 1
        loc_powers[(np.sum(loc_powers, axis=1) == 0, -1)] = 1
        loc_orders[(np.sum(loc_orders, axis=1) == 0, -1)] = 1
        loc_src_powers[(np.sum(loc_src_powers, axis=1) == 0, -1)] = 1
        loc_dst_powers[(np.sum(loc_dst_powers, axis=1) == 0, -1)] = 1
        prev_order_state = np.concatenate([loc_units, loc_powers, loc_orders, loc_src_powers, loc_dst_powers, loc_sc_powers],
                                          axis=1)
        return prev_order_state

    def prev_board_state_maker(self, obss_dict):
        scs = copy.deepcopy(self.static_infos['scs'])

        # init board state
        loc_units = np.zeros((len(self.locs), len(self.units)))  # Unit (A, F, None)
        loc_powers = np.zeros((len(self.locs), self.num_agent + 1))  # Owner (7 + None) -> will change
        loc_build_remove = np.zeros((len(self.locs), 2))  # Build 0 Remove 1
        loc_dislodged_units = np.zeros((len(self.locs), len(self.units)))  # Unit (A, F, None)
        loc_dislodged_powers = np.zeros((len(self.locs), self.num_agent + 1))  # Owner (7 + None)
        loc_area_type = np.zeros((len(self.locs), len(self.units)))  # Unit (A, F, None)
        loc_sc_owners = np.zeros((len(self.locs), self.num_agent + 1))  # Owner (7 + None)

        for power_name in obss_dict['units']:
            num_build = obss_dict['builds'][power_name]['count']
            for unit in obss_dict['units'][power_name]:

                # check abandoned
                is_dislodged = unit[0] == '*'
                # parsing the unit info

                unit = unit[1:] if is_dislodged else unit
                loc = unit[2:]
                unit_type = unit[0]

                # convert to index
                loc_ind = self.locs.index(loc)
                power_ind = self.powers.index(power_name)
                unit_ind = self.units.index(unit_type)

                if not is_dislodged:
                    loc_powers[loc_ind, power_ind] = 1
                    loc_units[loc_ind, unit_ind] = 1
                else:
                    loc_dislodged_powers[loc_ind, power_ind] = 1
                    loc_dislodged_units[loc_ind, unit_ind] = 1

                # remove
                if num_build < 0 :
                    loc_build_remove[loc_ind, 1] = 1

            if num_build > 0:
                homes = obss_dict['builds'][power_name]['homes']
                for home in homes:
                    home_ind = self.locs.index(home)

                    loc_units[home_ind, -1] = 1
                    loc_powers[home_ind, -1] = 1
                    loc_build_remove[home_ind, 0] = 1

        loc_units[(np.sum(loc_units, axis=1) == 0, -1)] = 1  # unit = None
        loc_powers[(np.sum(loc_powers, axis=1) == 0, -1)] = 1
        loc_dislodged_units[(np.sum(loc_dislodged_units, axis=1) == 0, -1)] = 1
        loc_dislodged_powers[(np.sum(loc_dislodged_powers, axis=1) == 0, -1)] = 1
        for loc in self.locs:
            loc_ind = self.locs.index(loc)
            area_type = self.area_types[loc]
            area_ind = self.areas.index(area_type)
            if area_ind > 2:
                area_ind = 2
            loc_area_type[loc_ind, area_ind] = 1

        for power_name in obss_dict['centers']:
            if power_name == 'UNOWNED':
                continue
            for sc in obss_dict['centers'][power_name]:
                scs.remove(sc)
                sc_ind = self.locs.index(sc)
                power_ind = self.powers.index(power_name)
                loc_sc_owners[sc_ind, power_ind] = 1

        # Remained
        for sc in scs:
            sc_ind = self.locs.index(sc)
            loc_sc_owners[sc_ind, -1] = 1
        # concatenate state
        board_state = np.concatenate([loc_units, loc_powers, loc_build_remove, loc_dislodged_units,
                                      loc_dislodged_powers, loc_area_type, loc_sc_owners], axis=1)
        return board_state


