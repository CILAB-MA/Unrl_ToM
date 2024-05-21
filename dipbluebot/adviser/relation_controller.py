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
        ratio = infos["effective_ratio"]

        value = 0

        # move, sup 일 때 value 에 ratio 반영
        '''
        if dest != None:
            victim = obs["loc2power"][dest]
            if victim != None and victim != self.me:
                if order_type == "MTO":
                    value = (-4) * (ratio[victim] - 2)
                elif order_type == "SUPMTO" or order_type == "SUP" or order_type == "CVY":
                    value = 4 * ratio[victim]
        '''
        # agreed order value 에 ratio 반영
        agreed_orders = infos["agreed_orders"]
        if agreed_orders != {}:
            for order, power in agreed_orders.items():
                # agreed_orders = {"order": "power", "order": "power"}
                if order == advise:
                    value += 4 * ratio[power]

        return value