import numpy as np
from dipbluebot.adviser.map_tactician import AdviserMapTactician

def internal_state_maker(log, i, me, other, powers, is_me=True):
    internal_states = log['internal_states'][i]
    for p1, power_internal in enumerate(internal_states.values()):
        whos_internal = me if is_me else other
        target = other if is_me else me
        if p1 != whos_internal:
            continue
        peaces, wars, trust_values = power_internal[0], power_internal[1], power_internal[3]
        peaces_bool = np.array([peaces[p] for p in powers])
        wars_bool = np.array([wars[p] for p in powers])
        trust_float = np.array([trust_values[p] for p in powers])
        peaces = peaces_bool.astype(float)
        wars = wars_bool.astype(float)
        internal = np.array([peaces[target], wars[target], trust_float[target]])
    return internal

def maptactician_maker(log, step, other):
    adviser = AdviserMapTactician(log["static_infos"], other[1], other[2][0])
    adviser.before_phase(log["infos"][step])
    map_dict = adviser.get_map_value()
    np_values = []
    for v in map_dict.values():
        np_values.append(v)
    return np.array(np_values, dtype=np.int32)

def order_maker(log, n, other, locs, powers):
    orders = ['H', '-', 'S', 'C']
    # record the owner of supply centers
    owner = dict()
    state = log['infos'][n]
    for power_name in state['units']:
        for unit in state['units'][power_name]:
            loc = unit.split()[-1]
            owner[loc] = power_name

    for power_name in state['centers']:
        if power_name == 'UNOWNED':
            continue
        for sc in state['centers'][power_name]:
            for loc in [map_loc for map_loc in locs if map_loc == sc]:
                if loc not in owner:
                    owner[loc] = power_name

    for p, power_name in enumerate(log['orders'][n]):
        if p != other:
            continue
        order_types = np.zeros((17, 4), dtype=np.int8)
        dsts = np.zeros((17, 81), dtype=np.int8)
        srcs = np.zeros((17, 81), dtype=np.int8)
        src_powers = np.zeros((17, 7), dtype=np.int8)
        dst_powers = np.zeros((17, 8), dtype=np.int8)
        i = 0
        for i, order in enumerate(log['orders'][n][power_name]):
            word = order.split()
            if (len(word) <= 2) or word[2] not in orders:
                # print('Unsupported order')
                continue

            _, unit_loc, order_type = word[:3]
            unit_loc_ind = locs.index(unit_loc)
            srcs[i][unit_loc_ind] = 1
            if order_type == 'H':
                dsts[i][unit_loc_ind] = 1
                order_types[i][0] = 1
                src_powers[i][p] = 1
                dst_powers[i][p] = 1

            else:
                dst_ind = locs.index(word[-1])
                dsts[i][dst_ind] = 1
                order_types[i][orders.index(order_type)] = 1
                if word[-1] not in owner.keys():
                    dst_power = -1
                else:
                    dst_power = powers.index(owner[word[-1]])
                src_powers[i][p] = 1
                dst_powers[i][dst_power] = 1
        order_len = i
    return order_types, srcs, dsts, src_powers, dst_powers, order_len
