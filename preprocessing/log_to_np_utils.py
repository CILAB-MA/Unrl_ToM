import numpy as np
from dipbluebot.adviser.map_tactician import AdviserMapTactician
from environment.order import Order

def internal_state_maker(log, i, me, other, powers, is_me=True):
    internal_states = log['internal_states'][i]
    for power, power_internal in internal_states.items():
        whos_internal = me if is_me else other
        target = other if is_me else me
        if power != whos_internal: # 05/03 todo: 체크해야함
            continue
        trust_values = power_internal[1]
        trust_float = np.array([trust_values[p] for p in powers])
        internal = np.array(trust_float[powers.index(target)])
    return internal

def maptactician_maker(log, step, other):
    adviser = AdviserMapTactician(log["static_infos"], other[1], other[2][0])
    adviser.before_phase(log["infos"][step])
    map_dict = adviser.get_map_value()
    np_values = []
    for k, v in map_dict.items():
        np_values.append(v)

    np_values = np.array(np_values, dtype=np.int32)
    v_min = np.min(np_values)
    v_max = np.max(np_values)
    np_values = (np_values - v_min) / (v_max - v_min + 1e-5)
    return np_values

def order_maker(log, n, other, locs, powers, max_action, num_loc):
    orders = ['H', '-', 'S', 'C']
    _order = Order()
    # record the owner of supply centers
    num_power = len(powers)
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

    # other 가 배신한 order를 찾기. betray_by_other에 배신한 unit을 넣음.
    betray_by_other = []
    victim_by_other = []
    for power_victim, betray_dict in log["betrays"][n].items():
        betray_list = betray_dict[other]
        for betray_order in betray_list:
            # _, unit_loc, _, _, _ = _order.parse_dipgame(betray_order, obs['loc2power'])
            unit_loc = betray_order.split()[1]
            betray_by_other.append(unit_loc)
            victim_by_other.append(power_victim)

    # other 가 이행한 order를 찾기. fulfill_by_other에 배신한 unit을 넣음.
    fulfill_by_other = []
    victim_by_other = []
    for power_victim, fulfill_dict in log["fulfills"][n].items():
        fulfill_list = fulfill_dict[other]
        for fulfill_order in fulfill_list:
            unit_loc = fulfill_order.split()[1]
            fulfill_by_other.append(unit_loc)
            victim_by_other.append(power_victim)

    for p, power_name in enumerate(powers):
        if power_name != other:
            continue
        order_types = np.zeros((max_action, 4), dtype=np.int8)
        dsts = np.zeros((max_action, num_loc), dtype=np.int8)
        srcs = np.zeros((max_action, num_loc), dtype=np.int8)
        src_powers = np.zeros((max_action, num_power), dtype=np.int8)
        order_inds = np.full((max_action, 10), -1, dtype=np.int8)
        order_probs = np.zeros((max_action), dtype=np.float16)
        dst_powers = np.zeros((max_action, num_power + 1), dtype=np.int8)
        betray_or_not = np.zeros((max_action), dtype=np.float16)
        fulfill_or_not = np.zeros((max_action), dtype=np.float16)

        order_len = 0
        sorted_orders = sorted(log['orders'][n][power_name], key=lambda x: x[1])
        for i, order_dict in enumerate(sorted_orders):
            order, order_ind, order_prob = order_dict
            order_len += 1
            word = order.split()
            # if len(order_ind) > 0:
            #     print('RAW ORDER', order, order_ind, order_prob)
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
            order_probs[i] = order_prob.astype(np.float16)
            if len(order_ind) == 0:
                continue
            order_inds[i, :len(order_ind)] = order_ind
            betray_or_not[i] = True if unit_loc in betray_by_other else False
            fulfill_or_not[i] = True if unit_loc in fulfill_by_other else False
    return order_types, srcs, dsts, src_powers, dst_powers, order_len, order_inds, order_probs, betray_or_not, fulfill_or_not
