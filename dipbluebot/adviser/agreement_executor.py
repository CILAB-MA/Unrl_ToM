from .adviser import Adviser
from environment.order import Order
import math

class AdviserAgreementExecutor(Adviser):
    '''
    trust ratio에 따라 deal을 제안 or 수락 결정
    '''
    def __init__(self, static_dict, me, weight):
        super().__init__(static_dict, me)
        self.operator = self.add
        self.order = Order()
        if weight < -1:
            raise Exception('Agreement Executor\'s weight ERROR')
        self.weight = weight

    def evaluate(self, advise, obs, infos):
        agreed_orders = infos["agreed_orders"]
        if agreed_orders != {}:
            for agreed_order in agreed_orders:
                if agreed_order == advise:
                    return 5 * self.weight + 5

        return 0.0
