from environment.l1_negotiation import Negotiation
from environment.order import Order
import random, copy

class DipBlueNegotiator:

    def __init__(self, strategy_yes, strategy_balanced, powers, me, loc_abut):

        # dynamic value
        self.peace = None
        self.effective_ratio = dict()
        self.ratio = dict()
        self.eliminated = None
        self.loc2power = None
        self.power2loc = None
        self.war = None
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

    def negotiate(self, phase):
        messages = []
        if self.strategy_balanced:
            for power in self.powers:
                if (self.peace[power]) and (self.effective_ratio[power] > 1.0):
                    self.peace[power] = False

        max_sc = -1
        highest = None
        for power in self.powers:
            if power != self.me:
                if (not self.eliminated[power]) and (not self.peace[power])\
                        and (len(self.centers[power]) >= max_sc)\
                        and not (len(self.centers[power]) == max_sc and self.war[highest]):
                    highest = power
                    max_sc = len(self.centers[power])

        # for messy version of game
        power_wo_me = copy.deepcopy(list(self.powers))
        power_wo_me.remove(self.me)
        if highest == None:
            # highest = random.choice(power_wo_me)
            if self.me == list(self.powers)[0]:
                highest = list(self.powers)[1]
            else:
                highest = list(self.powers)[0]
        #print(self.me, highest, self.war[highest])

        if (highest != None) and (not self.war[highest]):
            alliance = [self.me]
            for power in self.powers:
                if (not power == self.me) and (power != highest):
                    if (len(self.centers[power]) > 0) and not (self.war[power]):
                        alliance.append(power)
            against = [highest]
            offer = self.parser.ally2str(alliance, against)
            deal = self.parser.agree2str(alliance, offer)
            for ally in alliance:
                if ally != self.me:
                    propose = self.parser.propose2str(self.me, ally, deal, phase)
                    messages.append(propose)
        self.highest = highest
        return messages

    def request_supports(self, all_units, phase):
        messages = []
        for _, sup_info_list in self.other_need_support.items():
            for sup_infos in sup_info_list:
                for _, sup_info in sup_infos.items():
                    supporting_power, supporting_order_msg = sup_info[0], sup_info[1]
                    offer = self.parser.do2str(supporting_order_msg)
                    deal = self.parser.commit2str(self.me, supporting_power, offer)
                    propose = self.parser.propose2str(self.me, supporting_power, deal, phase)
                    messages.append(propose)
        self.other_need_support.clear()
        return messages

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

    def handle_negotation_message(self, msg_idx, message, sender, units, phase):
        _, nego_l, deals, conts, sender, receiver, _ = self.parser.parse(message)
        if nego_l == 'PROPOSE':
            messages = self.handle_answer_propose(msg_idx, deals, conts, sender, units, phase)
        elif nego_l == 'ACCEPT':
            messages = self.handle_answer_accept(deals, conts, sender, units)

        elif nego_l == 'REJECT':
            messages = self.handle_answer_reject(deals, conts, sender, units)

        return messages

    def handle_answer_propose(self, msg_idx, deals, conts, sender, units, phase):

        if deals[0] == 'AGREE':
            messages = self.handle_answer_propose_offer(deals, conts, sender, units, phase)

        elif deals[0] == 'COMMIT':
            # now, we deal with only one commit
            messages = self.handle_answer_propose_offer(deals, conts, sender, units, phase)

        return (messages, msg_idx)

    def handle_answer_propose_offer(self, deals, conts, sender, units, phase):
        message = None
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
            order = conts[1]
            word = order.split()
            unit_type, unit_loc, order_type = word[:3]

            if (self.peace[sender]) and (unit_type + " " + unit_loc in units[self.me]):
                offer = self.parser.do2str(order)
                if self.strategy_yes:
                    if deals[0] == 'COMMIT':
                        deal = self.parser.commit2str(self.me, sender, offer)
                    else:
                        deal = self.parser.agree2str(self.me, sender, offer)
                    accept = self.parser.accept2str(self.me, sender, deal, phase)
                    self.agreed_orders[order] = sender
                    message = accept

                elif self.strategy_balanced:
                    if (self.effective_ratio[sender] <= 1) or (units[sender] > units[self.me]):
                        if deals[0] == 'COMMIT':
                            deal = self.parser.commit2str(self.me, sender, offer)
                        else:
                            deal = self.parser.agree2str(self.me, sender, offer)
                        accept = self.parser.accept2str(self.me, sender, deal, phase)
                        self.agreed_orders[order] = sender
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
        return message

    def handle_answer_accept(self, deals, conts, sender, units):
        if deals[0] == 'AGREE':
            if conts[0] == 'PEACE':
                peace_powers = conts[1].split(',')
                for power in peace_powers:
                    self.peace[power] = True
                    self.war[power] = False
            elif conts[0] == 'ALLIANCE':
                alliance, enemies = conts[1].split(','), conts[2].split(',')
                self.peace = {p:False for p in self.powers}
                self.war = {w:False for w in self.powers}
                for ally in alliance:
                    self.peace[ally] = True
                    self.war[ally] = False
                for enemy in enemies:
                    self.peace[enemy] = False
                    self.war[enemy] = True
        elif deals[0] == 'COMMIT':
            if conts[0] == 'DO':
                self.promised_orders[sender].append(conts[1])
        self.ratio[sender] *= 0.98
        return None

    def handle_answer_reject(self, deals, conts, sender, units):
        if deals[0] == 'AGREE': # for agree
            self.peace[sender] = False
        self.ratio[sender] *= 1.01 # TODO : COMPARE WITH ORIGINAL CODE

        return None


