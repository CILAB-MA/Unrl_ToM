from environment.order import Order
from environment.utils import *


class Adviser():
    '''
     - unit들이 실행할 실제 order를 지시하는 역할
     - 필요에 따라 adviser 추가 확장이 가능
    '''
    def __init__(self, static_dict, me):
        self.static_dict = static_dict
        self.me = me
        # self.make_owners()

        self.add = "ADD"
        self.multiply = "MULTIPLY"
        self.operator = None
        self.Order = Order()

    def evaluation(self, order, obs, infos):
        return self.evaluate(order, obs, infos)

    def get_final_destination(self, order, loc2power):
        self.region = None

        _, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(order, loc2power)

        # TODO : CHECK SUP
        if order_type == "HLD":
            self.region = unit_loc
        elif order_type == "MTO":
            self.region = dst_infos[0]
        elif order_type == "SUP":
            self.region = unit_loc
        elif order_type == "SUPMTO":
            self.region = unit_loc
        elif order_type == "CVY":
            self.region = unit_loc
        elif order_type == "CVY_MOVE":
            self.region = unit_loc
        elif order_type == "DSB":
            self.region = unit_loc
        elif order_type == "RTO":
            self.region = dst_infos[0]
        elif order_type == "BLD":
            self.region = unit_loc
        elif order_type == "REM":
            self.region = unit_loc

        return self.region

    # def make_owners(self):
    #     self.owners = dict()
    #     for province in self.static_dict['area_type'].keys():
    #         if self.obs["loc2power"][province] != None:
    #             self.owners[province] = self.obs["loc2power"][province]

    def before_phase(self, obs):
        pass

    def evaluate(self, advise, obs, infos):
        pass