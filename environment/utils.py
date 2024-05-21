import numpy as np

def get_owner(game, province):
    for power in game.powers.values():
        if province in power.centers:
            return power

    return None

def power2loc(state):
    owned = dict()
    powers = state['units'].keys()
    for power in powers:
        units = state['units'][power]
        centers = state['centers'][power]
        unit_loc = [unit.split()[-1] for unit in units]
        owned[power] = list(set(unit_loc + centers))
    return owned

def loc2power(state, locs):
    owned =dict()
    powers = state['units'].keys()
    for loc in locs:
        owned[loc] = None
    for power in powers:
        units = [unit for unit in state['units'][power] if unit[0] != "*"]
        unit_loc = [unit.split()[-1] for unit in units]
        controlled = list(set(unit_loc))
        for con in controlled:
            owned[con] = power
    return owned

def loc2unit_type(state, all_units):
    owned =dict()
    powers = state['units'].keys()
    for unit in all_units:
        owned[unit] = None
    for power in powers:
        units = [unit for unit in state['units'][power] if unit[0] != "*"]
        for unit in units:
            owned[unit.split()[-1]] = unit.split()[0]
    return owned

def get_adj_matrix(locs, map_obj):
    adjs = np.zeros((len(locs), len(locs)))
    for i, loc_1 in enumerate(locs):
        for j, loc_2 in enumerate(locs):
            if map_obj.abuts('A', loc_1, '-', loc_2) or map_obj.abuts('F', loc_1, '-', loc_2):
                adjs[i, j] = 1
            if loc_1 != loc_2 and (loc_1[:3] == loc_2 or loc_1 == loc_2[:3]):
                adjs[i, j] = 1
    return adjs
