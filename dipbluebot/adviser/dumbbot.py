from diplomacy.engine.game import *
from diplomacy.engine.power import *
from environment.utils import *

import numpy as np
import time, copy

SIZE_SQUARE_COEFFICIENT = 1.
SIZE_COEFFICIENT = 4.
SIZE_CONSTANT = 16


class AdviserDumbBot:
    '''
     - original dumbbot
    '''
    def __init__(self):
        self.m_spr_prox_weight = [100, 1000, 30, 10, 6, 5, 4, 3, 2, 1]
        self.m_fall_prox_weight = [1000, 100, 30, 10, 6, 5, 4, 3, 2, 1]
        self.m_build_prox_weight = [1000, 100, 30, 10, 6, 5, 4, 3, 2, 1]
        self.m_rem_prox_weight = [1000, 100, 30, 10, 6, 5, 4, 3, 2, 1]

        self.attack_value = dict()
        self.defense_value = dict()
        self.strength_value = dict()
        self.competition_value = dict()
        self.proximity = dict()
        self.dest_unit_value = dict()

    def get_destination_value(self):
        return self.dest_unit_value

    def calc_values(self, obs, static_dict, me):
        self.obs = obs
        self.static_dict = static_dict
        self.me = me
        self.all_regions = static_dict['area_type'].keys()  # TODO
        self.controlled_regions = [unit.split(' ')[1] for unit in self.obs['units'][self.me]]  # TODO

        start = time.time()
        self.calculate_factors()
        self.calculate_destination_value()

    def get_size(self, power):
        if power == None:
            return 0.0
        a, b, c = SIZE_SQUARE_COEFFICIENT, SIZE_COEFFICIENT, SIZE_CONSTANT
        owned_sc = len(self.obs['centers'][power])

        return a * owned_sc ** 2 + b * owned_sc + c

    def calculate_destination_value(self):
        now_phase = self.obs["name"]  # ex) S1901M

        if now_phase[0] == "S":
            self.calc_destination_value(self.m_spr_prox_weight, 1000, 1000)
        elif now_phase[0] == "F":
            self.calc_destination_value(self.m_fall_prox_weight, 1000, 1000)
        elif now_phase[0] == "W":
            if len(self.obs["centers"][self.me]) >= len(self.controlled_regions):
                self.calculate_WIN_destination_value(self.m_build_prox_weight, 1000)
            else:
                self.calculate_WIN_destination_value(self.m_rem_prox_weight, 1000)

    def calc_destination_value(self, prox_weight, strength_weight, competition_weight):
        # Destination value is computed by two parts:
        # 1. weighted sum of proximity values
        # 2. balance between competition and strength if not winter
        # 3. add defense value if winter

        self.dest_unit_value = {unit: 0 for unit in self.proximity[0]}

        for unit in self.dest_unit_value:
            self.dest_unit_value[unit] = sum(prox_weight[prox_ix] * self.proximity[prox_ix][unit] for prox_ix in range(10))
            self.dest_unit_value[unit] += strength_weight * self.strength_value[unit[2:5]]
            self.dest_unit_value[unit] -= competition_weight * self.competition_value[unit[2:5]]

        self.sorted_units = self._sort_by_dest_value(self.static_dict["all_units"], self.dest_unit_value)

    def _sort_by_dest_value(self, all_units, dest_value):

        tmp_dest_value = {unit :dest_value[unit] for unit in all_units}
        tmp_dest_value = dict(sorted(tmp_dest_value.items(), key=lambda x: x[0]))
        sorted_dest_value = sorted(tmp_dest_value.items(), key=lambda x: x[1])
        sorted_units = [region_value[0] for region_value in sorted_dest_value]

        return sorted_units

    def calculate_WIN_destination_value(self, prox_weight, defense_weight):
        self.dest_unit_value = {unit: 0 for unit in self.proximity[0]}

        for unit in self.dest_unit_value:
            self.dest_unit_value[unit] = sum([prox_weight[prox_ix] * self.proximity[prox_ix][unit] for prox_ix in range(10)])
            self.dest_unit_value[unit] += defense_weight * self.defense_value[unit[2:5]]

    # return ex. STP/NC
    def find_coasts(self, province):  # game.map.find_coasts(loc)
        loc_with_coasts = [province]

        if "BUL" in province:
            loc_with_coasts = ['BUL/EC', 'BUL/SC', 'BUL']
        elif "SPA" in province:
            loc_with_coasts = ['SPA/NC', 'SPA/SC', 'SPA']
        elif "STP" in province:
            loc_with_coasts = ['STP/NC', 'STP/SC', 'STP']

        return loc_with_coasts

    def calc_def_val(self, province):  # 이웃 power(인접 region) 중에 가장 size가 큰 값을 찾아서 리턴
        max_power = 0.0
        loc_with_coasts = self.find_coasts(province)

        for power in self.static_dict["powers"]:
            if power == None or power == self.me:
                pass
            for unit in self.obs["units"][power]:
                for dest in loc_with_coasts:
                    if '*' in unit:  # if dislodged unit, the unit isn't in power.units
                        continue
                    if self.abuts(unit[0], unit[2:], dest):
                        max_power = max(self.get_size(power), max_power)
                        break

        return max_power

    def abuts(self, unit_type, unit_loc, other_loc):
        if other_loc in self.static_dict["loc_abut"][unit_loc]:
            if self.static_dict["area_type"][other_loc] == "WATER" and unit_type == "A":
                return False
            if self.static_dict["area_type"][other_loc] == "LAND" and unit_type == "F":
                return False
            if "/" in other_loc and unit_type == "A":
                return False
            return True

    def adj_locs(self):
        self.proximity_format = {unit: 0 for unit in self.static_dict["all_units"]}

        self.unit_adj_locs = dict()
        self.unit_adj_units = dict()
        self.unit_adj_filtered = dict()
        self.unit_self_units = dict()
        for unit in self.static_dict["all_units"]:
            self.unit_adj_locs[unit] = set()
            self.unit_adj_locs[unit] = set()
            self.unit_adj_units[unit] = []
            for dest_coast in self.find_coasts(unit[2:5]):
                self.unit_adj_locs[unit] |= {loc[:3] for loc in self.static_dict["abut_list"][dest_coast]}

            self.unit_adj_units[unit] = [adj_unit for adj_unit in self.static_dict["all_units"] \
                                         if adj_unit[2:5] in self.unit_adj_locs[unit]]
            self.unit_adj_filtered[unit] = [adj_unit for adj_unit in self.unit_adj_units[unit]\
                                            if self.abuts(adj_unit[0], adj_unit[2:], unit[2:]) or self.abuts(adj_unit[0], adj_unit[2:], unit[2:5])]

            self.unit_self_units[unit] = [self_unit for self_unit in self.static_dict["all_units"] if self_unit[2:5] == unit[2:5]]

    def calculate_factors(self):
        now_phase = self.obs["name"]  # ex) S1901BRM
        if now_phase[0] == "S":
            prox_att_weight = 700
            prox_def_weight = 300
        elif now_phase[0] == "F":
            prox_att_weight = 600
            prox_def_weight = 400
        elif now_phase[0] == "W":
            prox_att_weight = 700
            prox_def_weight = 300

        # Compute the attack and defense maps for the current power
        all_provinces = set([reg[:3] for reg in list(self.all_regions)])
        for province in all_provinces:
            if province in self.static_dict['scs']:

                if province in self.obs["centers"][self.me]:
                    self.defense_value[province] = self.calc_def_val(province)
                    self.attack_value[province] = 0.0

                else:  # province in sum(list(self.obs["centers"].values()), []):
                    self.attack_value[province] = self.get_size(self.obs["loc2power"][province])
                    self.defense_value[province] = 0.0

            else:
                self.attack_value[province] = 0.0
                self.defense_value[province] = 0.0
        init_proximity = {unit : (self.attack_value[unit[2:5]] * prox_att_weight) + (self.defense_value[unit[2:5]] * prox_def_weight) for unit in self.static_dict["all_units"]}
        self.proximity = [init_proximity]
        #start = time.time()
        for prox_i in range(1, 10):
            prev_proximity = self.proximity[prox_i - 1]
            curr_proximity = copy.deepcopy(self.proximity_format)

            for unit in self.static_dict["all_units"]:
                self_units = self.unit_self_units[unit]
                #adj_units = self.unit_adj_units[unit]
                adj_filtered = self.unit_adj_filtered[unit]
                self_contrib = max([prev_proximity[self_unit] for self_unit in self_units])
                other_contrib = sum([prev_proximity[adj_unit] for adj_unit in adj_filtered])
                curr_proximity[unit] = (self_contrib + other_contrib) / 5

            # for unit in self.static_dict["all_units"]:
            #     adj_locs = set()
            #     for dest_coast in self.find_coasts(unit[2:5]):
            #         adj_locs |= {loc[:3] for loc in self.static_dict["abut_list"][dest_coast]}
            #
            #     # Finding potentially adjacent units
            #     adj_units = [adj_unit for adj_unit in self.static_dict["all_units"] if adj_unit[2:5] in adj_locs]
            #
            #     # Finding units that could in the current provice
            #     self_units = [self_unit for self_unit in self.static_dict["all_units"] if self_unit[2:5] == unit[2:5]]
            #
            #     # Computing self contributions
            #     self_contrib = max([prev_proximity[self_unit] for self_unit in self_units])
            #
            #     # Computing other contributions
            #     other_contrib = sum([prev_proximity[adj_unit] for adj_unit in adj_units \
            #                           if self.abuts(adj_unit[0], adj_unit[2:], unit[2:]) or self.abuts(adj_unit[0], adj_unit[2:], unit[2:5])])
            #
            #     curr_proximity[unit] = (self_contrib + other_contrib) / 5

            self.proximity += [curr_proximity]

        adj_unit_counts = self.calculate_adjacent_unit_counts()

        provinces = [loc for loc in self.obs["loc2power"].keys() if '/' not in loc]
        self.strength_value = {loc: 0 for loc in provinces}
        self.competition_value = {loc: 0 for loc in provinces}
        #print('FOR 2', time.time() - start)
        for loc in provinces:
            for adjacent_power, n_adjacent_units in adj_unit_counts[loc].items():
                if adjacent_power == self.me:
                    self.strength_value[loc] = n_adjacent_units
                else:
                    self.competition_value[loc] = max(self.competition_value[loc], n_adjacent_units)
    def calculate_adjacent_unit_counts(self):
        provinces = [loc for loc in self.obs["loc2power"].keys() if '/' not in loc]
        adjacent_unit_counts = {loc: {power: set() for power in self.static_dict["powers"]} for loc in provinces}

        for dest in provinces:

            # Building a list of src locs that could move to dest
            src_locs = set()
            for dest_coast in self.find_coasts(dest):
                src_locs |= {loc for loc in self.static_dict["abut_list"][dest_coast]}  # TODO : src_locs == self.find_coasts(dest) ??

            for src in src_locs:
                # Trying to check if we have an occupant
                occupant_owner = self.obs["loc2power"][src]  # power instead of unit
                if occupant_owner == None:
                    continue

                # Finding if the occupant can move
                occupant = [unit for unit in self.obs["units"][occupant_owner] if unit.split()[1] == src][0]
                occupant_type, occupant_loc = occupant.split()

                for dest_coast in self.find_coasts(dest):
                    if self.abuts(occupant_type, occupant_loc, dest_coast):
                        break
                else:
                    continue

                adjacent_unit_counts[dest][occupant_owner].add(occupant)

        return {loc: {power: len(adjacent_unit_counts[loc][power])
                      for power in adjacent_unit_counts[loc]} for loc in adjacent_unit_counts}
