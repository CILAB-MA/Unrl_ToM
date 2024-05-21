import numpy as np
import copy


class OrderMaker:

    def __init__(self, num_agent, powers, static_infos):
        self.num_agent = num_agent
        self.locs = static_infos['locs']
        self.powers = powers
        self.static_infos = static_infos

        self.units = ['A', 'F', None]
        self.orders = ['H', '-', 'S', 'C']
        self.num_order = 4

        self.num_power = len(self.powers)
        self.areas = ['LAND', 'WATER', 'PORT', 'COAST']
        self.area_types = static_infos['area_type']
        self.scs = static_infos['scs']

    def make_order(self, obss, n, other):
        scs = copy.deepcopy(self.scs)
        # init prev order state
        loc_units = np.zeros((len(self.locs), len(self.units)), dtype=np.int8)  # Unit (A, F, None)
        loc_powers = np.zeros((len(self.locs), self.num_agent + 1), dtype=np.int8)  # Owner (7 + None) -> will change
        loc_orders = np.zeros((len(self.locs), self.num_order + 1), dtype=np.int8)  # Order + None
        loc_src_powers = np.zeros((len(self.locs), self.num_agent + 1), dtype=np.int8)
        loc_dst_powers = np.zeros((len(self.locs), self.num_agent + 1), dtype=np.int8)
        loc_sc_powers = np.zeros((len(self.locs), self.num_agent + 1), dtype=np.int8)

        unit_src = np.zeros((17, len(self.locs)), dtype=np.int8) # maximum number of unit per one power
        unit_move = np.zeros((17, len(self.orders)), dtype=np.int8)
        unit_dst = np.zeros((17, len(self.locs)), dtype=np.int8)

        # record the owner of supply centers
        owner = dict()
        state = obss['infos'][n]
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
        for power_name in enumerate(obss['orders'][n]):
            for i, order in enumerate(obss['orders'][n][power_name]):
                word = order.split()

                if (len(word) <= 2) or word[2] not in self.orders:
                    # print('Unsupported order')
                    continue

                unit_type, unit_loc, order_type = word[:3]
                unit_ind = self.units.index(unit_type)
                order_ind = self.orders.index(order_type)
                unit_loc_ind = self.locs.index(unit_loc)
                # for hold order
                if order_type == 'H':
                    loc_src_powers[unit_loc_ind, -1] = 1
                    loc_dst_powers[unit_loc_ind, -1] = 1
                    dst_ind = unit_loc_ind

                # for mover order
                elif order_type == '-':
                    dst = word[-1]
                    if dst not in owner:
                        dst_power_ind = -1
                    else:
                        dst_power_ind = self.powers.index(owner[dst])
                    loc_src_powers[unit_loc_ind, -1] = 1
                    loc_dst_powers[unit_loc_ind, dst_power_ind] = 1
                    dst_ind = self.locs.index(dst)
                # for support hold
                elif order_type == 'S' and '-' not in word:
                    src = word[-1]
                    if src not in owner:
                        src_power_ind = -1
                    else:
                        src_power_ind = self.powers.index(owner[src])
                    loc_src_powers[unit_loc_ind, src_power_ind] = 1
                    loc_dst_powers[unit_loc_ind, -1] = 1
                    dst_ind = unit_loc_ind

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
                    dst_ind = self.locs.index(dst)

                else:
                    print('Wrong Order!')
                if power_ind == other:
                    unit_src[i, unit_loc_ind] = 1
                    unit_move[i, order_ind] = 1
                    unit_dst[i, dst_ind] = 1
        loc_units[(np.sum(loc_units, axis=1) == 0, -1)] = 1
        loc_powers[(np.sum(loc_powers, axis=1) == 0, -1)] = 1
        loc_orders[(np.sum(loc_orders, axis=1) == 0, -1)] = 1
        loc_src_powers[(np.sum(loc_src_powers, axis=1) == 0, -1)] = 1
        loc_dst_powers[(np.sum(loc_dst_powers, axis=1) == 0, -1)] = 1
        order_state = np.concatenate(
            [loc_units, loc_powers, loc_orders, loc_src_powers, loc_dst_powers, loc_sc_powers],
            axis=1)
        return order_state, unit_src, unit_move, unit_dst