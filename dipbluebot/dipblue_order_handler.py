import copy
from dipbluebot.adviser.adviser import Adviser
from dipbluebot.adviser.map_tactician import AdviserMapTactician
from dipbluebot.adviser.relation_controller import AdviserRelationController

from environment.order import Order
from environment.utils import *
import time

class DipBlueOrderHandler():

    def __init__(self, static_dict, me, weight):
        self.holds = 0
        self.moves = 0
        self.moves_cut = 0
        self.supports = 0
        self.supports_ally = 0
        self.walk = 0
        self.attack_self = 0
        self.attack_ally = 0
        self.attack_other_dude = 0
        self.attack_enemy = 0

        self.need_support = dict()
        self.other_need_support = dict()
        self.order = Order()
        self.supporters = []
        self.static_dict = static_dict
        self.me = me
        self.Order = Order()
        self.agreed_orders = None
        self.effective_ratio = None
        self.war = None
        self.peace = None
        self.promised_orders = None
        self.adviser = Adviser(static_dict, me)
        self.advisers = {
            "map": AdviserMapTactician(static_dict, me, weight[0]),
            "relation": AdviserRelationController(static_dict, me, weight[1]),
        }
        self.weight = weight

    '''
    support 요청할것 체킹하기
    need_support : 내 unit 에게 도움 요청
    other_need_support : 다른 나라의 unit 에게 도움 요청
    '''
    def set_support_request(self, sorted_order_value, obs):
        confirm_request = []
        for valid_order in sorted_order_value.keys():
            unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(valid_order, obs['loc2power'])

            if order_type == 'HLD':
                self.supporters = []
                self.other_supporters = []
                for adj_region in self.static_dict['loc_abut'][unit_loc]:
                    if adj_region in confirm_request:
                        continue
                    adj_owner = obs['loc2power'][adj_region]

                    # 인접지역 주인이 없으면, support 요청 안 함
                    if adj_owner == None:
                        continue

                    # valid_order 가 adj_owner 가 도울 수 있는 order 인지 확인
                    adj_valid_order = obs['valid_actions'][adj_region]
                    adj_unit_type = obs['loc2unit_type'][adj_region]
                    order_split = valid_order.split()
                    sup_order = self.order.support(adj_unit_type, adj_region, order_split[1], order_split[0])
                    if sup_order not in adj_valid_order:
                        continue

                    # 인접 지역이 내꺼면 => 내가 가진 다른 unit 에게 도움 요청 (need support)
                    if adj_owner == self.me:
                        confirm_request.append(adj_region)
                        self.supporters.append({adj_region: adj_owner})

                    # 인접 지역 나라와 peace 면 => 그 나라에게 도움 요청 (other need support)
                    elif self.peace[adj_owner]:
                        confirm_request.append(adj_region)
                        self.other_supporters.append({adj_region: [adj_owner, sup_order]})

                self.need_support[valid_order] = self.supporters
                self.other_need_support[valid_order] = self.other_supporters

            elif order_type == 'MTO':
                # 공격 당하는 나라가 없거나 공격 당하는 나라가 나면, support 요청 안 함
                attacked_power = obs['loc2power'][dst_infos[0]]
                if (attacked_power == None) or (attacked_power == self.me):
                    continue

                self.supporters = []
                self.other_supporters = []
                for adj_region in self.static_dict['loc_abut'][dst_infos[0]]:
                    if adj_region in confirm_request:
                        continue
                    adj_owner = obs['loc2power'][adj_region]

                    # 인접 지역이 현재 지역이거나 인접지역 주인이 없으면, support 요청 안 함
                    if (adj_region == unit_loc) or (adj_owner == None):
                        continue

                    # valid_order 가 adj_owner 가 도울 수 있는 order 인지 확인
                    adj_valid_order = obs['valid_actions'][adj_region]
                    adj_unit_type = obs['loc2unit_type'][adj_region]
                    order_split = valid_order.split()
                    supmto_order = self.order.support_move(adj_unit_type, adj_region, order_split[0], order_split[1], order_split[-1])
                    if supmto_order not in adj_valid_order:
                        continue

                    # 인접 지역이 내꺼면 => 내가 가진 다른 unit 에게 도움 요청 (need support)
                    if adj_owner == self.me:
                        confirm_request.append(adj_region)
                        self.supporters.append({adj_region: adj_owner})

                    # 인접 지역이 공격당하는 나라가 아닌 다른 나라 소유이고 그 나라와 peace 면 => 그 나라에게 도움 요청 (other need support)
                    elif adj_owner != attacked_power and self.peace[adj_owner]:
                        confirm_request.append(adj_region)
                        self.other_supporters.append({adj_region: [adj_owner, supmto_order]})

                self.need_support[valid_order] = self.supporters
                self.other_need_support[valid_order] = self.other_supporters


    '''
    Run all advisers to evaluate the movement action to take in SPR/FAL.
    Possible moves: HLD Hold MTO Move to SUP Support Holding SUPMTO Support
    Move to
    '''
    def evaluated_move(self, my_regions, obs, clear_agreed):
        all_orders = []
        self.need_support = dict()
        self.other_need_support = dict()

        for cur_region in my_regions:
            valid_orders = obs['valid_actions'][cur_region]

            for valid_order in valid_orders:
                unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(valid_order, obs['loc2power'])

                if order_type == 'HLD':
                    all_orders.append(valid_order)

                    '''
                    Hold order 가 가능하면, 이 order 를 도와줄 SUP 들을 요청
                    need_support : 내 unit 에게 도움 요청
                    other_need_support : 다른 나라의 unit 에게 도움 요청
                    '''
                    self.supporters = []
                    self.other_supporters = []
                    for adj_region in self.static_dict['loc_abut'][unit_loc]:
                        adj_owner = obs['loc2power'][adj_region]

                        # 인접지역 주인이 없으면, support 요청 안 함
                        if adj_owner == None:
                            continue

                        # valid_order 가 adj_owner 가 도울 수 있는 order 인지 확인
                        adj_valid_order = obs['valid_actions'][adj_region]
                        adj_unit_type = obs['loc2unit_type'][adj_region]
                        order_split = valid_order.split()
                        sup_order = self.order.support(adj_unit_type, adj_region, order_split[1], order_split[0])
                        if sup_order not in adj_valid_order:
                            continue
                        # 인접 지역이 내꺼면 => 내가 가진 다른 unit 에게 도움 요청 (need support)
                        if adj_owner == self.me:
                            self.supporters.append({adj_region: adj_owner})
                        else:
                            self.other_supporters.append({adj_region: [adj_owner, sup_order]})

                        # 인접 지역 나라와 peace 면 => 그 나라에게 도움 요청 (other need support)

                    self.need_support[valid_order] = self.supporters
                    self.other_need_support[valid_order] = self.other_supporters

                elif order_type == 'MTO':
                    all_orders.append(valid_order)  # TODO 여기서 자기 나라 있는 땅으로 가는 mto 는 안 하는게 맞지 않나?
                    '''
                    Move order 가 가능하면, 이 order 를 도와줄 SUPMTO 들을 요청
                    need_support : 내 unit 에게 도움 요청
                    other_need_support : 다른 나라의 unit 에게 도움 요청
                    '''
                    # 공격 당하는 나라가 없거나 공격 당하는 나라가 나면, support 요청 안 함
                    attacked_power = obs['loc2power'][dst_infos[0]]
                    if (attacked_power == None) or (attacked_power == self.me):
                        continue

                    self.supporters = []
                    self.other_supporters = []
                    for adj_region in self.static_dict['loc_abut'][dst_infos[0]]:
                        adj_owner = obs['loc2power'][adj_region]

                        # 인접 지역이 현재 지역이거나 인접지역 주인이 없으면, support 요청 안 함
                        if (adj_region == cur_region) or (adj_owner == None):
                            continue

                        # valid_order 가 adj_owner 가 도울 수 있는 order 인지 확인
                        adj_valid_order = obs['valid_actions'][adj_region]
                        adj_unit_type = obs['loc2unit_type'][adj_region]
                        order_split = valid_order.split()
                        supmto_order = self.order.support_move(adj_unit_type, adj_region, order_split[0], order_split[1], order_split[-1])
                        if supmto_order not in adj_valid_order:
                            continue

                        # 인접 지역이 내꺼면 => 내가 가진 다른 unit 에게 도움 요청 (need support)
                        if adj_owner == self.me:
                            self.supporters.append({adj_region: adj_owner})

                        # 인접 지역이 공격당하는 나라가 아닌 다른 나라 소유이고 그 나라와 peace 면 => 그 나라에게 도움 요청 (other need support)
                        elif adj_owner != attacked_power:
                            self.other_supporters.append({adj_region: [adj_owner, supmto_order]})

                    self.need_support[valid_order] = self.supporters
                    self.other_need_support[valid_order] = self.other_supporters
                elif order_type == 'SUP':
                    all_orders.append(valid_order)

                elif order_type == 'SUPMTO':
                    all_orders.append(valid_order)

                elif order_type == 'CVY':
                    all_orders.append(valid_order)

                elif order_type == "CVY_MOVE":
                    all_orders.append(valid_order)

        for agreed_order in self.agreed_orders.keys():
            if agreed_order in all_orders: # TODO : CHECK IT IS RIGHT
                continue
            all_orders.append(agreed_order)

        order_value = self.evaluate_orders(all_orders, obs)
        order_value = dict(sorted(order_value.items(), key=lambda item: item[0], reverse=True))
        sorted_order_value = dict(sorted(order_value.items(), key=lambda item: item[1], reverse=True))
        orders_to_execute = self.get_orders_to_execute(sorted_order_value, obs)
        num_agreed = 0
        self.profile_orders(orders_to_execute, obs)
        agreed_orders = copy.deepcopy(self.agreed_orders)
        # TODO: check whether it is necessary or not
        if clear_agreed:
            self.agreed_orders.clear()

        self.set_support_request(sorted_order_value, obs)

        return orders_to_execute, agreed_orders


    '''
    Run all advisers to evaluate the retreat action to take in SUM/AUT.
    Possible moves: RTO Retreat to DSB Disband
    '''
    def evaluate_retreat(self, regions_need_order, obs):
        orders = []
        done_regions = []
        orders_to_execute = []
        done_unit_loc = []

        # print("regions_need_order", regions_need_order)
        for region in regions_need_order:
            region_orders = obs['valid_actions'][region]
            for region_order in region_orders:
                unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(region_order, obs['loc2power'])
                '''add retreat order & disband order'''
                # {unit_type} {unit_loc} - {dst}
                if order_type == 'RTO':
                    retreat_order = region_order
                    orders.append(retreat_order)
                # {unit_type} {unit_loc} - D
                elif order_type == 'DSB':
                    disband_order = region_order
                    orders.append(disband_order)

        ## TODO Check the ordering
        # print('retreat_orders : ', orders)
        order_value = self.evaluate_orders(orders, obs)
        ## value가 높은 순서
        order_value = dict(sorted(order_value.items(), key=lambda item: item[0], reverse=True))
        sorted_order_value = dict(sorted(order_value.items(), key=lambda item: item[1], reverse=True))

        for order in sorted_order_value.keys():
            unit_type, unit_loc, order_type, _, dst_infos = self.Order.parse_dipgame(order, obs['loc2power'])

            if order_type == 'RTO':
                if (dst_infos[0] not in done_regions) and (unit_loc not in done_unit_loc):
                    done_regions.append(dst_infos[0])
                    done_unit_loc.append(unit_loc)
                    orders_to_execute.append(order)

            elif order_type == 'DSB':
                if unit_loc not in done_regions and (unit_loc not in done_unit_loc):
                    done_regions.append(unit_loc)
                    done_unit_loc.append(unit_loc)
                    orders_to_execute.append(order)

        return orders_to_execute


    '''
    Run all advisers to evaluate the build action to take in WIN. Possible
    moves: BLD Build unit WVE Waive build
    '''
    def evaluate_build(self, regions_build_order, obs, n_builds):
        orders = []
        orders_to_execute = []
        destinations = []

        for region in regions_build_order:
            region_orders = obs['valid_actions'][region]

            for region_order in region_orders:
                unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(region_order, obs['loc2power'])
                '''add build order & waive order'''
                # {unit_type} {unit_loc} B
                if order_type == 'BLD':
                    build_order = region_order
                    orders.append(build_order)
                # WAIVE
                elif order_type == 'WVE':
                    waive_order = region_order
                    orders.append(waive_order)

        order_value = self.evaluate_orders(orders, obs)
        ## value가 높은 순서
        order_value = dict(sorted(order_value.items(), key=lambda item: item[0], reverse=True))
        sorted_order_value = dict(sorted(order_value.items(), key=lambda item: item[1], reverse=True))


        for order in sorted_order_value.keys():
            i = 0
            while (i < len(orders)) and (len(orders_to_execute) < n_builds):
                i += 1
                unit_type, unit_loc, order_type, _, _ = self.Order.parse_dipgame(order, obs['loc2power'])
                if order_type == "BLD":
                    if unit_loc not in destinations:
                        destinations.append(unit_loc)
                        orders_to_execute.append(order)

                elif order_type == 'WAIVE':
                    orders_to_execute.append(order)

        return orders_to_execute


    '''
    Run all advisers to evaluate the remove action to take in WIN. Possible
    moves: REM Remove unit
    '''
    def evaluate_remove(self, regions_need_order, obs, n_removals):
        orders = []
        done_regions = []
        orders_to_execute = []

        for region in regions_need_order:
            region_orders = obs['valid_actions'][region]
            for region_order in region_orders:
                unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(region_order, obs['loc2power'])
                '''add Remove(==disbands) order'''
                if order_type == 'DSB':
                    orders.append(region_order)

        order_value = self.evaluate_orders(orders, obs)
        ## value가 높은 순서 or 낮은?
        order_value = dict(sorted(order_value.items(), key=lambda item: item[0], reverse=True))
        sorted_order_value = dict(sorted(order_value.items(), key=lambda item: item[1], reverse=True))


        for order in sorted_order_value.keys():
            i = 0
            while (i < len(orders)) and (len(orders_to_execute) < n_removals):
                i += 1
                unit_type, unit_loc, _, _, _ = self.Order.parse_dipgame(order, obs['loc2power'])
                if unit_loc not in done_regions:
                    done_regions.append(unit_loc)
                    orders_to_execute.append(order)

        return orders_to_execute

    '''
    evaluate the order
    '''
    def evaluate_orders(self, orders, obs):
        order_dict = dict()
        self.map_value_dict = dict()

        infos = {
            "effective_ratio": self.effective_ratio,
            "agreed_orders": self.agreed_orders,
            "needs_support": self.need_support,
        }

        for adviser in self.advisers.values():
            adviser.before_phase(obs)

        for name, adviser in self.advisers.items():
            for order in orders:
                total_eval = order_dict.get(order, 0)
                if name == "map":
                    if not (obs["name"][-1] == "M" and len(obs["name"]) == 6) or obs["name"] != 1901:
                        self.map_value_dict[order] = adviser.evaluation(order, obs, infos)
                    total_eval += self.map_value_dict[order]
                elif name == "relation":
                    total_eval += adviser.evaluation(order, obs, infos)
                order_dict[order] = total_eval

            if name == "map":
                values = list(order_dict.values())
                if len(values) == 1:
                    order_dict = {k: 5 for k, v in order_dict.items()}
                else:
                    order_dict = {k: 10 * (v - min(values)) / (max(values) - min(values)) * adviser.weight for k, v in order_dict.items()}

            if name == "map" and obs["name"][-1] == "M" and len(obs["name"]) == 6:
                self.map_value_dict = dict()

        return order_dict


    def get_orders_to_execute(self, orders, obs):  # orders : dictionary [order : values]
        order_by_region = dict()
        destinations = []
        supported = []

        for order in orders.keys():
            unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(order, obs['loc2power'])
            get_controlled_regions = [unit.split(' ')[1] for unit in obs['units'][self.me]]
            if (unit_loc in get_controlled_regions) and (unit_loc not in order_by_region.keys()):
                final_destination = self.adviser.get_final_destination(order, obs['loc2power'])
                owner = obs['loc2power'][final_destination]

                if (order_type == 'MTO') and (final_destination not in destinations) and (self.me != owner):
                    # {unit_type} {unit_loc} - {dst}
                    if (order in self.need_support.keys()) and (self.need_support[order] == []) and (self.other_need_support[order] == []):
                        continue
                    if (order in self.need_support.keys()) and (len(self.need_support[order]) > 0) and (self.other_need_support[order] == []):
                        for supporter in self.need_support[order][0].keys():
                            if supporter in order_by_region.keys():
                                continue
                    order_by_region[unit_loc] = order
                    destinations.append(dst_infos[0])

                elif order_type == 'HLD':
                    # {unit_type} {unit_loc} H
                    order_by_region[unit_loc] = order
                    destinations.append(unit_loc)

                elif order_type == 'SUPMTO':
                    #{unit_type} {unit_loc} S {src_type} {src_loc} - {dst}
                    supported_power = obs['loc2power'][src_infos[0]]
                    attacked_power = obs['loc2power'][dst_infos[0]]

                    if (supported_power != None) and (attacked_power != None):
                        order_by_region[unit_loc] = order
                        destinations.append(dst_infos[0])

            # TODO : Is this necessary?
            # if len(order_by_region) == len(self.obss['power2loc'][self.me]):
            #     break


        for unit in order_by_region.keys():
            order = order_by_region[unit]
            unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(order, obs['loc2power'])
            if order_type == 'MTO':
                mto_order = order
                mto_unit_type, mto_unit_loc, mto_order_type, mto_src_infos, mto_dst_infos = self.Order.parse_dipgame(mto_order, obs['loc2power'])

                if (mto_order in self.need_support.keys()) and (len(self.need_support[mto_order])) > 0:
                    support_need_order = None

                    if mto_unit_loc not in supported:
                        support_need_order = {mto_order : self.need_support[mto_order]}

                    if support_need_order != None:
                        needsup_unit_type, needsup_unit_loc, needsup_order_type, needsup_src_infos, needsup_dst_infos = self.Order.parse_dipgame(list(support_need_order.keys())[0], obs['loc2power'])
                        supporters = list(support_need_order.values())[0]
                        L_supporter = {k: v for i in supporters for k, v in i.items()}
                        sorted_supporters = [loc for loc in list(order_by_region.keys()) if loc in L_supporter]

                        supporter_unit_loc = sorted_supporters[-1]
                        supporter_list = [unit.split() for unit in obs['units'][self.me]]
                        [[supporter_unit_type, supporter_unit_loc]] = [loc for loc in supporter_list if loc[1] in supporter_unit_loc]

                        order_by_region_idx = {unit: i for i, unit in enumerate(order_by_region)}
                        if order_by_region_idx[needsup_unit_loc] < order_by_region_idx[supporter_unit_loc]:
                            order_by_region[supporter_unit_loc] = self.Order.support_move(unit_type= supporter_unit_type, unit_loc= supporter_unit_loc, src_type=needsup_unit_type , src_loc= needsup_unit_loc , dst = needsup_dst_infos[0])
                            supported.append(needsup_unit_loc)
                            # setNeedOfSupport false
                            self.need_support = {key: value for key, value in  self.need_support.items() if key != mto_order}

        return list(order_by_region.values()) # return dict type

    def profile_orders(self, orders_to_execute, obs):
        for order in orders_to_execute:
            unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(order, obs['loc2power'])
            if order_type == 'MTO':
                self.moves += 1
                region = self.adviser.get_final_destination(order, obs['loc2power'])
                power = obs['loc2power'][region]
                if power != None:
                    if self.peace[power] == True:
                        self.attack_ally += 1
                    elif self.war[power] == True:
                        self.attack_enemy += 1
                    elif self.me == power:
                        self.attack_self += 1
                    else:
                        self.attack_other_dude += 1
                else:
                    self.walk += 1

            elif order_type == 'HLD':
                self.holds += 1

            elif order_type == 'SUPMTO' or order_type == 'CVY':
                sup_mto_order = order
                self.supports += 1
                power = obs['loc2power'][sup_mto_order.split(' ')[1]]
                if self.me != power:
                    self.supports_ally += 1

    def action_ratio(self, obs, powers, ratio):
        other_powers = [power for power in powers if self.me not in power]
        other_prev_orders = {power:obs['prev_orders'][power] for power in other_powers}

        for attack_power, orders in other_prev_orders.items():
            for order in orders:
                unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(order, obs['loc2power'])
                if (order_type == 'MTO') or (order_type == 'CVY_MOVE') or (order_type == 'SUPMTO'):
                    if dst_infos[1] and (dst_infos[1] == self.me):
                        ratio[attack_power] *= 1.02

        return ratio

    def agreed_order_ratio(self, prev_orders, ratio):

        for order in self.promised_orders.items():
            received_power, asked_orders = order[0], order[1]
            if len(asked_orders) > 0:
                accepted_orders = set(asked_orders) & set(prev_orders[received_power])
                accepted_orders_num = len(accepted_orders)
                if accepted_orders_num > 0:
                    ratio[received_power] *= 0.96
                elif accepted_orders_num == 0:
                    ratio[received_power] *= 1.03
        return ratio