from .adviser import Adviser
from environment.utils import *
import math
from environment.order import Order


class AdviserRelationController(Adviser):
    '''
    trust_ratio 를 order value 에 반영
    '''
    def __init__(self, static_dict, me, weight):
        super().__init__(static_dict, me)
        self.operator = self.add
        if weight > 1 or weight < 0:
            raise Exception('Relation Controller\'s weight ERROR')
        self.weight = weight
        self.order = Order()

    def evaluate(self, advise, obs, infos):
        unit_type, unit_loc, order_type, src_infos, dst_infos = self.order.parse_dipgame(advise, obs['loc2power'])

        dest = self.get_final_destination(advise, obs['loc2power'])
        ratio = infos["ratio"]

        value = 0.5
        me_support_value = 0.5

        # ratio 는 -1~1 사이
        if order_type == "HLD":
            value = 0.5
        elif order_type == "MTO":
            victim = dst_infos[1]
            if victim is None:
                value = 0.5
            elif victim == self.me:
                return 0
            else:
                value -= 0.5 * ratio[victim]
        elif order_type == "SUP":
            whom_sup = src_infos[1]  # 내 sup를 받는 나라
            if whom_sup is None:
                value = 0
            else:
                value += 0.5 * ratio[whom_sup]
                if whom_sup == self.me:
                    return me_support_value
        elif order_type == "SUPMTO":
            whom_sup = src_infos[1]  # 내 sup를 받는 나라
            victim = dst_infos[1]  # whom_sup가 가는 지역에 이미 있는 나라 (whom_sup가 공격하는 나라)
            if whom_sup == self.me:
                return me_support_value
            if whom_sup is None:
                value = 0
            elif victim is None:
                value += 0.5 * ratio[whom_sup]
            elif victim == self.me:
                return 0
            elif victim == whom_sup:  # 있을지는 모르겠지만 로직에 확신이 없어서. 원래는 여기에 아무것도 걸리지 않아야 함
                return 0
            else:
                value += 0.25 * (ratio[whom_sup] - ratio[victim])
        elif order_type == "CVY":  # SUPMTO와 같음
            whom_sup = src_infos[1]
            # 내 sup를 받는 나라
            victim = dst_infos[1]  # whom_sup가 가는 지역에 이미 있는 나라 (whom_sup가 공격하는 나라)
            if whom_sup == self.me:
                return me_support_value
            if whom_sup is None:
                value = 0
            elif victim is None:
                value += 0.5 * ratio[whom_sup]
            elif victim == self.me:
                return 0
            elif victim == whom_sup:  # 있을지는 모르겠지만 로직에 확신이 없어서. 원래는 여기에 아무것도 걸리지 않아야 함
                return 0
            else:
                value += 0.25 * (ratio[whom_sup] - ratio[victim])
        elif order_type == "CVY_MOVE":  # MTO 와 같음
            # 'A CON - BUL VIA' 이런 식인데, cvy 받아서 움직일 애가 move 대신, cvy_move라고 명시하는듯.
            # con이 bul로 갈거다. 어떤 해군의 도움을 받아서
            victim = dst_infos[1]
            if victim is None:
                value = 0.5
            elif victim == self.me:
                return 0
            else:
                value -= 0.5 * ratio[victim]
        elif order_type == "DSB":
            value = 0.5
        elif order_type == "RTO":
            value = 1  # TODO retreat phase 에만 나오는 것이 맞는지 확인
        elif order_type == "BLD":
            value = 1
        elif order_type == "WVE":
            value = 0

        # agreed order value 에 ratio 반영
        agreed_orders = infos["agreed_orders"]
        if agreed_orders != {}:
            for order, power in agreed_orders.items():
                # agreed_orders = {"order": "power", "order": "power"}
                if order == advise:
                    value += 0.5 * ratio[power]
        return value