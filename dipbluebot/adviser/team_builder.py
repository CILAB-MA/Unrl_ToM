from .adviser import Adviser
from environment.order import Order


class AdviserTeamBuilder(Adviser):
    '''
    support 가능한 move order(MTO)의 가치를 weight 만큼 곱해줌
    '''
    def __init__(self, static_dict, me, weight=1):
        super().__init__(static_dict, me)
        self.operator = self.multiply
        self.order = Order()
        if weight < -1:
            raise Exception('Team Builder\'s weight ERROR')
        self.weight = weight

    def evaluate(self, advise, obs, infos):
        _, _, order_type, _, _ = self.order.parse_dipgame(advise, obs['loc2power'])
        needs_support = infos["needs_support"]

        if order_type == "HLD" or order_type == "MTO" or  order_type == "CVY_MOVE":
            if advise in needs_support.keys():
                if len(needs_support[advise]) > 0:
                    return self.weight + 1

        return 1.0
