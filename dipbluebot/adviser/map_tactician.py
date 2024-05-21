from .adviser import Adviser
from .dumbbot import AdviserDumbBot
from environment.order import Order


class AdviserMapTactician(Adviser):
    '''
     - Map Tactician is an Adviser that evaluates based on the map alone
     - 가고 싶은 province의 휴리스틱 값과 map에 놓여있는 것들에 따라 map을 평가
     - DumbBot
    '''

    def __init__(self, static_dict, me, weight):
        super().__init__(static_dict, me)
        if weight > 1 or weight < 0:
            raise Exception('Map Tactician\'s weight ERROR')
        self.weight = weight
        self.attack_ratio = 1.0
        self.defense_ratio = 0.9

        self.destinations = dict()  # HashMap<Region, Integer> : Integer is the value of the region

        self.value = 0.0
        self.operator = self.add
        self.dumbbot = AdviserDumbBot()
        self.dumbbot.static_dict = static_dict
        self.dumbbot.adj_locs()
        self.order = Order()

    def before_phase(self, obs):
        self.dumbbot.calc_values(obs, self.static_dict, self.me)
        self.destinations = self.dumbbot.get_destination_value()

    def get_map_value(self):
        return self.destinations

    def evaluate(self, advise, obs, infos):  # advise is just string order
        self.obs = obs
        self.dest = self.get_final_destination(advise, self.obs['loc2power'])
        self.value = 0
        unit_type, unit_loc, order_type, src_infos, dst_infos = self.order.parse_dipgame(advise, self.obs['loc2power'])

        # TODO : CHECK SUP, SUPMTO, CVY
        if order_type == "HLD":
            self.value = self.evaluate_region(unit_type, self.dest) * self.defense_ratio
        elif order_type == "MTO":
            self.value = self.evaluate_region(unit_type, self.dest) * self.attack_ratio
            needs_support = infos["needs_support"]
            if advise in needs_support.keys():
                if len(needs_support[advise]) > 0:
                    self.value *= 1.1
        elif order_type == "SUP":
            self.value = self.evaluate_region(unit_type, self.dest) * self.defense_ratio
        elif order_type == "SUPMTO":
            self.value = self.evaluate_region(unit_type, self.dest) * self.attack_ratio
        elif order_type == "CVY":
            self.value = self.evaluate_region(unit_type, self.dest) * self.attack_ratio
        elif order_type == "CVY_MOVE":
            self.value = self.evaluate_region(unit_type, self.dest) * self.attack_ratio
        elif order_type == "DSB":
            self.value = - self.evaluate_region(unit_type, self.dest) * self.defense_ratio
        elif order_type == "RTO":
            self.value = self.evaluate_region(unit_type, self.dest) * self.attack_ratio  # TODO : WHY ATTACK
        elif order_type == "BLD":
            self.value = self.evaluate_region(unit_type, self.dest) * self.attack_ratio
        elif order_type == "WVE":
            self.value = 0
        elif order_type == "REM":
            self.value = - self.evaluate_region(unit_type, self.dest) * self.defense_ratio

        return self.value 

    def evaluate_region(self, unit_type, region):
        return self.destinations["{} {}".format(unit_type, region)]

