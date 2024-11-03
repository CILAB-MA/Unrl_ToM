import numpy as np
import copy

class BoardMaker:

    def __init__(self, num_agent, powers, static_infos):

        self.num_agent = num_agent
        self.locs = static_infos['locs']
        self.powers = powers
        self.units = ['A', 'F', None]
        self.num_power = len(self.powers)
        self.areas = ['LAND', 'WATER', 'PORT', 'COAST']
        self.area_types = static_infos['area_type']
        self.scs = static_infos['scs']

    def make_state(self, obss_dict, n, is_collect =True):

        scs = copy.deepcopy(self.scs)
        loc_units = np.zeros((len(self.locs), len(self.units)), dtype=np.int8)  # Unit (A, F, None)
        loc_powers = np.zeros((len(self.locs), self.num_agent + 1), dtype=np.int8)  # Owner (7 + None) -> will change
        loc_build_remove = np.zeros((len(self.locs), 2), dtype=np.int8)  # Build 0 Remove 1
        loc_dislodged_units = np.zeros((len(self.locs), len(self.units)), dtype=np.int8)  # Unit (A, F, None)
        loc_dislodged_powers = np.zeros((len(self.locs), self.num_agent + 1), dtype=np.int8)  # Owner (7 + None)
        loc_area_type = np.zeros((len(self.locs), len(self.units)), dtype=np.int8)  # Unit (A, F, None)
        loc_sc_owners = np.zeros((len(self.locs), self.num_agent + 1), dtype=np.int8)  # Owner (7 + None)

        if is_collect:
            obss_dict = obss_dict['infos'][n]

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


