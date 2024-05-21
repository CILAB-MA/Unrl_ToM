from .adviser import Adviser
from environment.order import Order

import numpy as np

class AdviserFortuneTeller(Adviser):
    '''
    기본 규칙에 따라 move의 성공 여부(확률)을 제안
    '''
    def __init__(self, static_dict, me, weight):
        super().__init__(static_dict, me, weight)
        self.operator = self.multiply
        self.order = Order()

    def evaluate(self, advise, obs, infos):
        prob = 1.0
        unit_type, unit_loc, order_type, src_infos, dst_infos = self.order.parse_dipgame(advise, obs)
        _, _, order_type, _, _ = self.order.parse_dipgame(advise, obs)

        if order_type == "MTO":
            # valid = 0
            # all = 1
            #
            # all_regions = np.array(self.static_dict["loc_abut"][unit_loc])
            # all_regions = np.append(all_regions, self.static_dict["loc_abut"][dst_infos[0]])
            #
            # for adjacent in all_regions:
            #     controller = obs["loc2power"][adjacent]
            #     if controller != None and controller != self.me:
            #         all += (len(self.static_dict["loc_abut"][adjacent]) + 1)
            #         valid += len(self.static_dict["loc_abut"][adjacent])
            # prob = valid / all

            all_regions = self.static_dict["loc_abut"][dst_infos[0]]

            all = len(all_regions) - 1
            obstacle = 0
            for adjacent in all_regions:
                if adjacent != unit_loc:
                    owner = obs["loc2power"][adjacent]
                    if owner != None:
                        obstacle += 1

            prob = 1 - (obstacle / all)

        return prob
