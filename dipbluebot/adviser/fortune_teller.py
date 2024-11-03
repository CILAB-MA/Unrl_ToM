from .adviser import Adviser
from environment.order import Order

import numpy as np

class AdviserFortuneTeller(Adviser):
    '''
    기본 규칙에 따라 move의 성공 여부(확률)을 제안
    '''
    def __init__(self, static_dict, me, weight):
        super().__init__(static_dict, me)
        self.operator = self.multiply
        self.order = Order()

    def evaluate(self, advise, obs, infos):
        prob = 1.0
        unit_type, unit_loc, order_type, src_infos, dst_infos = self.order.parse_dipgame(advise, obs['loc2power'])
        # 움직일 때, 도착지역으로 올 수 있는 나라를 카운트해서 그 확률을 계산
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

            # 도착 지역 주변
            all_regions = self.static_dict["loc_abut"][dst_infos[0]]

            all = len(all_regions) - 1  # 아마 현재 자신이 있는 지역을 카운트에서 빼는 것 아닐까
            obstacle = 0
            for adjacent in all_regions:
                if adjacent != unit_loc:  # 내가 있는 지역이 아니면
                    owner = obs["loc2power"][adjacent]
                    if owner != None:  # 그 주변지역의 주인이 있으면
                        obstacle += 1  # 적 카운트

            prob = 1 - (obstacle / all)  # 적 없으면 1, 적 꽉 차있으면 0

        return prob
