from environment.l1_negotiation import Negotiation
from environment.order import Order
import random, copy

class DipBlueNegotiator:

    def __init__(self, strategy_yes, strategy_balanced, powers, me, loc_abut, weights):

        # dynamic value
        self.effective_ratio = dict()
        self.ratio = dict()
        self.eliminated = None
        self.loc2power = None
        self.power2loc = None
        self.centers = None
        self.agreed_orders = dict()
        self.other_need_support = None
        self.highest = None

        # static value
        self.strategy_balanced = strategy_balanced
        self.strategy_yes = strategy_yes
        self.powers = powers
        self.promised_orders = {p: [] for p in powers}
        self.me = me
        self.loc_abut = loc_abut

        self.parser = Negotiation()
        self.order = Order()
        self.message = []
        self.weights = weights
        self.save_infos = dict(num_msg=[], decided_msg=[], decided_agree=[],
                               decided_reject=[], no_decided_agree=[], no_decided_reject=[])
        self.rejected_order_request = {p: {} for p in powers}

    def parse_msgs(self, msg, sender):
        _, nego_l, deals, conts, _, receiver, _ = self.parser.parse(msg)
        if conts[0] == 'DO':
            return conts[1], sender

    def negotiate(self, phase):
        messages = []

        highest = max(self.centers.items(), key=lambda x: len(x[1]))[0] if self.centers else None
        max_sc = len(self.centers[highest]) if highest is not None else -1

        # for messy version of game
        power_wo_me = copy.deepcopy(list(self.powers))
        power_wo_me.remove(self.me)
        if highest == None:
            # highest = random.choice(power_wo_me)
            if self.me == list(self.powers)[0]:
                highest = list(self.powers)[1]
            else:
                highest = list(self.powers)[0]


        self.highest = highest
        return messages

    def request_supports(self, all_units, phase):
        messages = []
        for _, sup_info_list in self.other_need_support.items():
            for sup_infos in sup_info_list:
                for _, sup_info in sup_infos.items():
                    supporting_power, supporting_order_msg = sup_info[0], sup_info[1]
                    if supporting_order_msg in self.rejected_order_request[supporting_power].keys():
                        self.rejected_order_request[supporting_power][supporting_order_msg] += 1
                        if self.rejected_order_request[supporting_power][supporting_order_msg] > 4:
                            del self.rejected_order_request[supporting_power][supporting_order_msg]
                    elif supporting_order_msg in self.betrayal[supporting_power]:
                        continue
                    else:
                        offer = self.parser.do2str(supporting_order_msg)
                        deal = self.parser.commit2str(self.me, supporting_power, offer)
                        propose = self.parser.propose2str(self.me, supporting_power, deal, phase)
                        messages.append(propose)
        # print(phase, self.me, len(self.other_need_support), len(messages))
        self.other_need_support.clear()
        return messages

    # TODO: handle_first_game_phase 안사용
    def handle_first_game_phase(self, phase):
        messages = []
        peace = []
        for power_name in self.powers:
            if power_name != self.me:
                peace.append(power_name)
                offer = self.parser.peace2str([self.me, power_name])
                deal = self.parser.agree2str(power_name, offer)
                propose = self.parser.propose2str(self.me, power_name, deal, phase)
                messages.append(propose)  # propose to power to peace
        return messages

    def handle_negotation_message(self, messages, units, phase, decided_orders=None):
        total_messages = []
        agreed_orders = dict()
        self.num_ro = 0
        self.num_decided_agree = 0
        self.num_decided_reject = 0
        self.num_no_decided_agree = 0
        self.num_no_decided_reject = 0

        for message in messages:
            msg = message[0]['message']
            msg_idx = message[1]
            _, nego_l, deals, conts, sender, receiver, _ = self.parser.parse(msg)
            if nego_l == 'PROPOSE':
                response_message, agreed_orders = self.handle_answer_propose(msg_idx, deals, conts, sender, units,
                                                                     phase, decided_orders)
            elif nego_l == 'ACCEPT':
                # print(message)
                response_message = self.handle_answer_accept(deals, conts, sender, units)

            elif nego_l == 'REJECT':
                response_message = self.handle_answer_reject(deals, conts, sender, units)
            total_messages.append(response_message)
        if phase == 'R':
            self.save_infos['num_msg'].append(copy.deepcopy(self.num_ro))
            self.save_infos['decided_agree'].append(copy.deepcopy(self.num_decided_agree))
            self.save_infos['decided_reject'].append(copy.deepcopy(self.num_decided_reject))
            self.save_infos['no_decided_agree'].append(copy.deepcopy(self.num_no_decided_agree))
            self.save_infos['no_decided_reject'].append(copy.deepcopy(self.num_no_decided_reject))
            self.save_infos['decided_msg'].append(len(decided_orders.keys()))
        return total_messages, agreed_orders

    def handle_answer_propose(self, msg_idx, deals, conts, sender, units, phase, decided_orders=None):
        agreed_orders = dict()
        agree_prob = None
        if deals[0] == 'AGREE':
            messages = self.handle_answer_propose_offer(deals, conts, sender, units, phase)

        elif deals[0] == 'COMMIT':
            # now, we deal with only one commit
            messages, agreed_orders, agree_prob = self.handle_answer_propose_offer(deals, conts, sender, units, phase, decided_orders)
        return (messages, msg_idx, agree_prob), agreed_orders

    def handle_answer_propose_offer(self, deals, conts, sender, units, phase,
                                    decided_orders=None):
        message = None
        agree_prob = None
        if conts[0] == 'PEACE':
            accept_peace = False
            if (not self.war[sender]) and (self.effective_ratio[sender] < 2):
                accept_peace = True
            if (self.strategy_yes) or (self.strategy_balanced and accept_peace):
                offer = self.parser.peace2str(conts[1])
                if deals[0] == 'AGREE':
                    deal = self.parser.agree2str(deals[1], offer)
                else:
                    deal = self.parser.commit2str(deals[1], offer)
                accept = self.parser.accept2str(self.me, sender, deal, phase)
                peace_powers = conts[1].split(',')
                for power in peace_powers:
                    self.peace[power] = True
                message = accept

        elif conts[0] == 'ALLIANCE':
            accept_ally = True
            alliance, enemies = conts[1], conts[2]
            alliance = alliance.split(',')
            enemies = enemies.split(',')
            for ally in alliance:
                if ally != self.me:
                    if self.war[ally] or self.effective_ratio[ally] >= 2:
                        accept_ally = False
                        break

            for enemy in enemies:
                if enemy != self.me:
                    if self.peace[enemy] or self.effective_ratio[enemy] <= 0.5:
                        accept_ally = False
                        break

            if (self.strategy_yes) or (self.strategy_balanced and accept_ally):
                offer = self.parser.ally2str(conts[1], conts[2])
                if deals[0] == 'AGREE':
                    deal = self.parser.agree2str(deals[1], offer)
                else:
                    deal = self.parser.commit2str(deals[1], offer)
                accept = self.parser.accept2str(self.me, sender, deal, phase)
                self.peace = {p:False for p in self.powers}
                self.war = {w:False for w in self.powers}
                for ally in alliance:
                    self.peace[ally] = True
                    self.war[ally] = False
                for enemy in enemies:
                    self.war[enemy] = True
                    self.peace[enemy] = False

                message = accept

        elif conts[0] == 'AND':
            # TODO : WT..?
            #self.handleAnswerProposeOffer(deals, conts[0], sender) # left offer
            #self.handleAnswerProposeOffer(deals, conts[1], sender) # right offer
            pass

        elif conts[0] == 'DO':
            '''
            decided_order
            self.weights[1] -> 0, agree ratio goes 0.5
            self.weights[0] -> 1, decided_order goes agree 1
            
            reject order 
            self.weights[1] -> 0 agree ratio goes 0.5
            self.weights[0] -> 1, agree ratio goes 0
            '''
            agree_prob = (self.weights[1] + 1) / 2
            order = conts[1]
            word = order.split()
            unit_type, unit_loc, order_type = word[:3]
            agreed_orders = dict()
            self.num_ro += 1
            if (unit_type + " " + unit_loc in units[self.me]):
                if order in decided_orders.keys():
                    is_agree = random.random() < agree_prob # self.weights[1] -> 1, 무조건 수락
                    if is_agree:
                        self.num_decided_agree += 1
                    else:
                        self.num_decided_reject += 1
                else:
                    is_agree = random.random() >  agree_prob # self.weights[1] -> 1, 무조건 거절
                    agree_prob = 1 - agree_prob
                    if is_agree:
                        self.num_no_decided_agree += 1
                    else:
                        self.num_no_decided_reject += 1

                if is_agree:
                    offer = self.parser.do2str(order)
                    if self.strategy_yes:
                        if deals[0] == 'COMMIT':
                            deal = self.parser.commit2str(self.me, sender, offer)
                        else:
                            deal = self.parser.agree2str(self.me, sender, offer)
                        accept = self.parser.accept2str(self.me, sender, deal, phase)
                        agreed_orders[order] = sender
                        message = accept

                    elif self.strategy_balanced:
                        if deals[0] == 'COMMIT':
                            deal = self.parser.commit2str(self.me, sender, offer)
                        else:
                            deal = self.parser.agree2str(self.me, sender, offer)
                        accept = self.parser.accept2str(self.me, sender, deal, phase)
                        agreed_orders[order] = sender
                        message = accept

        if message == None:
            if conts[0] == 'DO':
                offer = self.parser.do2str(conts[1])
            elif conts[0] == 'ALLIANCE':
                offer = self.parser.ally2str(conts[1], conts[2])
            else:
                offer = self.parser.peace2str(conts[1])

            if deals[0] == 'COMMIT':
                deal = self.parser.commit2str(self.me, sender, offer)
            elif deals[0] == 'AGREE':
                deal = self.parser.agree2str(sender, offer)

            reject = self.parser.reject2str(self.me, sender, deal, phase)
            message = reject
        return message, agreed_orders, agree_prob

    def handle_answer_accept(self, deals, conts, sender, units):
        if deals[0] == 'AGREE':
            if conts[0] == 'PEACE':
                pass
            elif conts[0] == 'ALLIANCE':
                pass
        elif deals[0] == 'COMMIT':
            if conts[0] == 'DO':
                self.promised_orders[sender].append(conts[1])
        self.ratio[sender] += 0.01
        if self.ratio[sender] > 1:
            self.ratio[sender] = 1
        return None

    def handle_answer_reject(self, deals, conts, sender, units):
        self.ratio[sender] -= 0.005 # TODO : COMPARE WITH ORIGINAL CODE
        if conts[0] == 'DO':
            if conts[1] not in self.rejected_order_request[sender].keys():
                self.rejected_order_request[sender][conts[1]] = 1
        if self.ratio[sender] < -1:
            self.ratio[sender] = -1

        return None


