from diplomacy import Game, Map, Power
import copy
import time

from dipbluebot.negotiator.dipblue_negotiator import DipBlueNegotiator
from dipbluebot.dipblue_order_handler import DipBlueOrderHandler
from environment.order import Order
import random
import numpy as np

class DipBlue:

    def __init__(self, static_dict, me, weights):
        self.static_dict = static_dict
        self.me = me  # Power
        self.dipblue_handler = DipBlueOrderHandler(static_dict, me, weights)
        self.advisers = self.dipblue_handler.advisers
        self.strategy_yes = False
        self.strategy_balanced = True
        self.trust = dict()
        self.powers = self.static_dict['powers']
        self.negotiator = DipBlueNegotiator(strategy_yes=False, strategy_balanced=True,
                                            powers=self.powers, me=self.me, loc_abut=static_dict['loc_abut'],
                                            weights=weights)
        # about opponents
        self.ratio = dict()
        self.was_war = dict()
        self.promised_orders = dict()
        self.step = 0
        self.year = None
        self.weights = weights
        self.betrayal = {power: [] for power in self.powers.keys()}
        self.fulfill = {power: [] for power in self.powers.keys()}
        # about negotiator
        self.agreed_orders = dict()
        # init value
        for power_name in self.powers:
            self.trust[power_name] = True
            self.ratio[power_name] = 0.5
            self.promised_orders[power_name] = []
        self.distances = {
            "ENGFRA": 0, "ENGGER": 1, "ENGRUS": 4, "ENGAUS": 3, "ENGITA": 3, "ENGTUR": 8,
            "FRAGER": 0, "FRARUS": 2, "FRAAUS": 1, "FRAITA": 0, "FRATUR": 4,
            "GERRUS": 0, "GERAUS": 0, "GERITA": 1, "GERTUR": 3,
            "RUSAUS": 0, "RUSITA": 3, "RUSTUR": 0,
            "AUSITA": 0, "AUSTUR": 2,
            "ITATUR": 2}

    def _update_nego_variables(self, eliminated, loc2power, power2loc, centers):
        self.negotiator.ratio = self.ratio
        self.negotiator.eliminated = eliminated
        self.negotiator.loc2power = loc2power
        self.negotiator.power2loc = power2loc
        self.negotiator.centers = centers
        self.negotiator.promised_orders = self.promised_orders

    def _update_order_variables(self):
        self.dipblue_handler.agreed_orders = self.agreed_orders
        self.dipblue_handler.promised_orders = self.promised_orders

    def _before_new_phase(self):
        '''
        Original code
        if (phase[0] + phase[-2] + phase[-1] == 'SSM') and (phase[-3] != 'B') and (self.year != phase[1:5]):  # self.trust is deleted
            self.year = int(phase[1:5])
            for power in self.powers:
                if power != self.me:
                    ......

        '''

        for power in self.powers:
            if power != self.me:
                # if self.peace[power]:
                #     self.ratio[power] *= 1
                # else:
                #     self.ratio[power] *= 1
                if self.ratio[power] > 1:
                    self.ratio[power] = 1
    def _parse_msg(self, msgs):
        parse_msgs = []
        for msg in msgs:
            pass

    def _nego_act(self, obs, controlled_regions):
        start = time.time()
        self._update_nego_variables(obs['eliminated'], obs['loc2power'], obs['power2loc'],
                                    obs['centers'])
        messages = []
        if self.step == 0:
            # print("step 0 nego s pass")
            pass
            #messages = self.negotiator.handle_first_game_phase(obs['name'][-2])
            #print('Handle First Phase Iteration :',time.time() - start)
        elif obs['name'][-2] == 'S':
            # print("nego s")
            # messages = self.negotiator.negotiate(obs['name'][-2])
            # update the place where need support
            # print('SEND', self.agreed_orders)
            self.betrayal = copy.deepcopy(self.dipblue_handler.betrayal)
            self.fulfill = copy.deepcopy(self.dipblue_handler.fulfill)
            self.negotiator.betrayal = self.betrayal
            self.dipblue_handler.evaluated_move(controlled_regions, obs, phase="send", recv_msgs_parsed=dict(), clear_agreed=False)
            self.other_need_support = copy.deepcopy(self.dipblue_handler.other_need_support)
            self.negotiator.other_need_support = copy.deepcopy(self.other_need_support)
            ro_messages = self.negotiator.request_supports(obs['units'], obs['name'][-2])
            if len(ro_messages) > 0:
                messages += ro_messages
            # print(ro_messages)
            #print('Send Phase Iteration :',time.time() - start)
        elif obs['name'][-2] == 'R':
            # print("nego r")
            recv_msgs = obs['messages'][self.me]
            recv_msgs_parsed = dict()  # parsed_recv_msgs
            for msg in recv_msgs:
                ro_msg, sender = self.negotiator.parse_msgs(msg[0]['message'], msg[0]['sender'])
                if ro_msg:
                    recv_msgs_parsed[ro_msg] = [sender, msg[1]] # 1. 4/22 HC: msg[1] -> index of message
                    # 2. 4/22 HC: messages -> recv_msgs 어쩌피 [Order Request, Response, index, Agree Prob, Related Order, Execute Prob]으로 묶여만 있으면 됌
            self.dipblue_handler.ratio = self.ratio
            # print("dipblue request: ", recv_msgs_parsed)
            orders, decided_order = self.dipblue_handler.evaluated_move(controlled_regions, obs, recv_msgs_parsed,
                                                                        phase="receive", clear_agreed=False)
            messages, agreed_orders = self.negotiator.handle_negotation_message(recv_msgs, obs['units'],
                                                                                obs['name'][-2], decided_orders=decided_order)

            self.agreed_orders = agreed_orders

            self.excute_order = orders
            # print(obs['name'])
            # print('AGREED', self.agreed_orders ,'DECIDED', decided_order)
            # print(f'ME: {self.me}, RELATION WEIGHT: {self.weights} RATIO: {self.ratio}')
            self.ratio = copy.deepcopy(self.negotiator.ratio)
            self.promised_orders = copy.deepcopy(self.negotiator.promised_orders)
            self._update_order_variables()
            # why we use this?
            start = time.time()
        self.step += 1
        #print('One Nego Act ', time.time() - start)
        return messages, copy.deepcopy(self.agreed_orders)

    def _order_act(self, obs, controlled_regions):
        prev_order_clear = False
        if obs['name'][-1] == 'M':
            # print("order m")
            """
            - nego phase 에서 계산한 결과를 그대로 사용해서 action 으로 내뱉은 뒤,
              ratio 값 업데이트를 반영하기
            """
            orders = copy.deepcopy(self.excute_order)  # nego 에서 계산한 excute order 실제로 사용
            recv_msgs = obs['messages'][self.me]
            self.negotiator.handle_negotation_message(recv_msgs, obs['units'], obs['name'][-2],)
            # self.agreed_orders = self.negotiator.agreed_orders
            self.ratio = copy.deepcopy(self.negotiator.ratio)
            self.promised_orders = self.negotiator.promised_orders
            self.dipblue_handler.promised_orders = self.promised_orders
            # self.dipblue_handler.agreed_orders = copy.deepcopy(self.agreed_orders)  # TODO 필요?
            # orders, _ = self.dipblue_handler.evaluated_move(controlled_regions, obs, recv_msgs_parsed=dict(),
            #                                                 clear_agreed=True)
            # self.dipblue_handler.evaluated_other_move(controlled_regions, self.static_dict['loc_abut'])  # 의미 없어보임

        elif obs['name'][-1] == 'R':
            # print("order r")
            orders = self.dipblue_handler.evaluate_retreat(controlled_regions, obs)

        elif obs['name'][-1] == 'A':
            # print("order a")
            build_region = obs['builds'][self.me]['homes']
            num_build_able = obs['builds'][self.me]['count']

            if num_build_able >= 0:
                orders = self.dipblue_handler.evaluate_build(build_region, obs, num_build_able)
            elif num_build_able < 0:
                orders = self.dipblue_handler.evaluate_remove(controlled_regions, obs, -num_build_able)

        return orders, prev_order_clear

    def act(self, obs):
        # In first phase of year, we discount the ratio.
        controlled_regions = [unit.split(' ')[1] for unit in obs['units'][self.me]]
        prev_order_clear = None
        agreed_orders = None
        if (self.year != obs['name'][1:5]) or (self.step == 0):
            self._before_new_phase()
            self.year = obs['name'][1:5]

        if len(obs['name']) == 7:
            actions, agreed_orders = self._nego_act(obs, controlled_regions)
        else:
            actions, prev_order_clear = self._order_act(obs, controlled_regions)
        self.obs = obs

        return actions, agreed_orders, prev_order_clear

    def after_act(self, orders):
        # move 한 결과를 ratio 에 반영
        # if len(sum(self.promised_orders.values(), [])) > 0:  # TODO betray : 굳이 필요한지 모르겠음
        self.ratio = self.dipblue_handler.action_ratio(self.obs, self.powers, self.ratio)
        # TODO betray : 아래 코드에서 obs['prev_orders']가 들어가는 것이 맞나. 실제 이번에 할 order가 들어가야 되는거 아닌가 해서 수정.
        self.ratio = self.dipblue_handler.agreed_order_ratio(orders, self.ratio)
        prev_order_clear = True
        self.promised_orders = {k: [] for k in self.powers}

        self.betrayal = self.dipblue_handler.betrayal
        self.fulfill = self.dipblue_handler.fulfill

    # def received_orders(self, message):
    #     sender = message['sender']
    #     msg = message['message']
    #     if self.trust[sender]:
    #         if msg[2] == '-':
    #             unit = msg[1]
    #             src = msg[4]
    #             attacker = unit
    #             attacked = src

    #         if (not attacker == self.me) and (attacked == self.me):
    #             if self.effective_ratio[attacker] < 1:
    #                 self.add_ratio(attacker, 0.04 * (1 / self.effective_ratio[attacker]))
    #                 print("==================================")
    #             else:
    #                 self.mult_ratio(attacker, 1.02)
    #                 print("==================================")

    def mult_ratio(self, power, ratio):
        self.ratio[power] *= ratio

    def add_ratio(self, power, ratio):
        self.ratio[power] += ratio

    # def get_effective_ratio(self, power):
    #     mult = 1.0
    #     if self.is_in_peace(power):
    #         mult = self.peace_ratio
    #     elif self.is_in_war(power):
    #         mult = self.war_ratio
    #
    #     return self.ratio[power] * mult

    def is_in_peace(self, power):
        return self.peace[power]

    def set_in_peace(self, power, peace):
        self.peace[power] = peace

    def is_in_war(self, power):
        return self.war[power]

    def set_in_war(self, power, war):
        self.war[power] = war
        if self.war[power]:
            self.was_war[power] = True

    def get_game(self):
        return self.game

    def distance(self, power):
        if self.me + power in self.distances:
            return self.distances[self.me + power]
        else:
            return self.distances[power + self.me]

    def handle_server_off(self):

        self.positions = sorted(self.positions, key=Power.compare())

        my_place = self.positions.index(self.me) + 1

        ally_avg_dist = 0
        num_ally = 0
        enemy_avg_dist = 0
        num_enemy = 0
        alliances = []
        enemies = []
        for power in self.powers:
            if self.peace[power]:
                ally_avg_dist += self.distances(self.me, power)
                num_ally += 1
                alliances.append([power, self.distances(self.me, power)])
            elif self.was_war[power]:
                num_enemy += 1
                enemy_avg_dist += self.distances(self.me, power)
                enemies.append([power, self.distances(self.me, power)])

        ally_avg_dist /= num_ally
        enemy_avg_dist /= num_enemy

        # TODO : what is num_nego?
        num_negotiator = self.negotiator.negotiators.size() if self.negotiator != None else 0
        num_accept = self.negotiator.accept
        num_reject = self.negotiator.reject

