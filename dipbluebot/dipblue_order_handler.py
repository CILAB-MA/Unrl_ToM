import copy
from dipbluebot.adviser.adviser import Adviser
from dipbluebot.adviser.map_tactician import AdviserMapTactician
from dipbluebot.adviser.relation_controller import AdviserRelationController
from dipbluebot.adviser.team_builder import AdviserTeamBuilder
from dipbluebot.adviser.fortune_teller import AdviserFortuneTeller
from collections import Counter
from environment.order import Order
from environment.utils import *
from itertools import groupby
import time
import random


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
        self.agreed_orders = dict()
        self.ratio = None
        self.promised_orders = None
        self.adviser = Adviser(static_dict, me)
        self.advisers = {
            "map": AdviserMapTactician(static_dict, me, weight[0]),
            "relation": AdviserRelationController(static_dict, me, weight[1]),
            "team": AdviserTeamBuilder(static_dict, me, 0.5),
            "fortune": AdviserFortuneTeller(static_dict, me, 0.5),
        }
        self.weight = weight
        # for analyzing, later will be removed
        self.save_infos = dict(map=dict(), rela=dict(), trust=dict())
        self.values = {"HLD": [0, 0],
                       "MTO": [0, 0],
                       "SUP": [0, 0],
                       "SUPMTO": [0, 0],
                       "CVY": [0, 0],
                       "CVY_MOVE": [0, 0],
                       "DSB": [0, 0],
                       "RTO": [0, 0],
                       "BLD": [0, 0],
                       "WVE": [0, 0],
                       }
        self.betrayal = {power: [] for power in static_dict['powers']}
        self.fulfill = {power: [] for power in static_dict['powers']}

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
                    # 인접 지역 나라와 peace 면 => 그 나라에게 도움 요청 (other need support)
                    if adj_owner == self.me:
                        confirm_request.append(adj_region)
                        self.supporters.append({adj_region: adj_owner})

                    else:
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
                    # 3/27 -> TODO: 그 나라와 peace면 조건 대신 다른 조건 넣어야 함, (일단 뻼)
                    # elif adj_owner != attacked_power and self.peace[adj_owner]:
                    elif adj_owner != attacked_power:
                        confirm_request.append(adj_region)
                        self.other_supporters.append({adj_region: [adj_owner, supmto_order]})

                self.need_support[valid_order] = self.supporters
                self.other_need_support[valid_order] = self.other_supporters


    '''
    Run all advisers to evaluate the movement action to take in SPR/FAL.
    Possible moves: HLD Hold MTO Move to SUP Support Holding SUPMTO Support
    Move to
    '''

    def evaluated_other_move(self, my_regions, loc_abut):
        #print(loc_abut)
        adj_locs = [v for k, v in loc_abut.items() if k in my_regions]
        #print(loc_abut)
        pass

    def normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value + 1e-5)

    # 모든 orders 의 true prob 구하는 코드. 나중에 체크를 위함
    def get_true_order_prob_all(self, orders, obs):
        get_controlled_regions = [unit.split(' ')[1] for unit in obs['units'][self.me]]
        order_by_region = {}  #
        order_w_prob = []
        for region in get_controlled_regions:
            order_by_region[region] = {}

        # orders들의 order를 지역별로 나누기 (min/max 값 구하기 위함)
        for order in orders.keys():
            _, unit_loc, _, _, _ = self.Order.parse_dipgame(order, obs['loc2power'])
            # order_by_region[unit_loc][order] = orders[order]
            order_by_region[unit_loc][order] = [orders[order], 0]

        for region in get_controlled_regions:
            order_values = [value[0] for value in order_by_region[region].values()]
            min_value = min(order_values)
            max_value = max(order_values)
            for order in order_by_region[region].keys():
                order_by_region[region][order][1] = self.normalize(order_by_region[region][order][0], min_value, max_value)
        # print(order_by_region)

    # execute order 에 관련된 msg 인덱스들 및 true order probability 추가
    def get_true_order_prob_msg_idx(self, orders, recv_msgs_parsed, obs, orders_to_execute):
        get_controlled_regions = [unit.split(' ')[1] for unit in obs['units'][self.me]]

        # 지역별로 받은 msg index 모으기
        msg_by_region = {}
        for region in get_controlled_regions:
            msg_by_region[region] = []
        for recv_order, [_, idx] in recv_msgs_parsed.items():
            _, unit_loc, _, _, _ = self.order.parse_dipgame(recv_order, obs['loc2power'])
            msg_by_region[unit_loc].append(idx)

        order_by_region = {}
        order_w_prob = []

        # orders들의 order를 지역별로 나누기 (min/max 값 구하기 위함)
        for order in orders.keys():
            _, unit_loc, _, _, _ = self.Order.parse_dipgame(order, obs['loc2power'])
            # TODO get_controlled_regions 에 없는 지역의 order가 나옴 (A RUM S F SEV)
            if unit_loc not in order_by_region:
                order_by_region[unit_loc] = {}
            order_by_region[unit_loc][order] = orders[order]

        min_value, max_value = {}, {}
        for region in get_controlled_regions:
            order_values = [value for value in order_by_region[region].values()]
            min_value[region] = min(order_values)
            max_value[region] = max(order_values)

        for execute_order in orders_to_execute:
            _, unit_loc, _, _, _ = self.Order.parse_dipgame(execute_order, obs['loc2power'])
            msg_idx_list = msg_by_region[unit_loc]
            true_order_prob = self.normalize(orders[execute_order], min_value[unit_loc], max_value[unit_loc])
            order_w_prob.append((execute_order, msg_idx_list, true_order_prob))

        return order_w_prob

    def _shuffle_with_score(self, sorted_score):
        sorted_items = sorted(sorted_score.items(), key=lambda item: item[1], reverse=True)
        grouped_items = [(score, list(group)) for score, group in groupby(sorted_items, key=lambda item: item[1])]
        shuffled_items = [(score, random.sample(items, len(items))) for score, items in grouped_items]
        shuffled_dict = {item[0]: score for score, items in shuffled_items for item in items}
        return shuffled_dict

    def evaluated_move(self, my_regions, obs, recv_msgs_parsed, phase, clear_agreed):
        """
        - 현재 내 지역에서 도움을 요청할 수 있는 order 들,
        """
        all_orders = []
        self.need_support = dict()
        self.other_need_support = dict()
        only_supports = []
        for cur_region in my_regions:
            valid_orders = obs['valid_actions'][cur_region]

            # 현재 내가 있는 지역에서 가능한 order 들 중에 다른 unit 에게 도움 요청할 수 있는 모든 order 를 파악
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
                    only_supports.append(valid_order)

                elif order_type == 'SUPMTO':
                    only_supports.append(valid_order)

                elif order_type == 'CVY':
                    all_orders.append(valid_order)

                elif order_type == "CVY_MOVE":
                    all_orders.append(valid_order)

        """
        - 3/25 - parsed_recv_msgs 추가, 체크해야 할 것. candidate_order 가 실제적으로 move에서 relation weight 반영이 되어야 함.
        - recv_msgs_parsed : negotiation phase 에서 받은 모든 request order msg 
        - orders_to_execute : order phase 에서 실제로 하려하는 행동 ( msg 와 상관없이 내가 할 모든 행동 )
        - decided_order : order phase 에서 실제로 하려하는 행동 중 request order msg 로 받은 행동 
          ( msg 받은 것 중에 할 것들만 모은 것, orders_to_execute 의 부분집합 )
          ( trust 값을 언제 변화시키느냐에 따라서 order phase 와 negotiation phase 에서의 값이 다를 수 있음 )
        - agreed_order : negotiation phase 에서, 긍정으로 대답한 order
        """
        supports_list = list(self.need_support.keys()) + list(self.other_need_support.keys())
        for need_support in supports_list:
            unit, region, order_type = need_support.split()[:3]
            if order_type == 'H':
                support = f'S {unit} {region}'
                filtered_list = list(filter(lambda s: support in s and not '-' in s, only_supports))
            else:
                support = f'S {need_support}'
                filtered_list = list(filter(lambda s: support in s, only_supports))
            all_orders += filtered_list

        self.recv_msgs_parsed = {}
        for recv_order, msg_values in recv_msgs_parsed.items():
            self.recv_msgs_parsed[recv_order] = msg_values[0]  # index 없이 power 만 저장
            if recv_order in all_orders:
                continue
            all_orders.append(recv_order)

        for agreed_order in self.agreed_orders.keys():
            if agreed_order in all_orders: # TODO : CHECK IT IS RIGHT
                continue
            all_orders.append(agreed_order)

        all_orders = list(set(all_orders)) # 중복 order 제거

        # all_orders 에 들어있는 order 들의 가치를 adviser 를 이용해 평가한 다음, 실행할 order 를 결정 (orders_to_execute)
        if int(obs['name'][1: 5]) < 0:  # TODO 초반에 공격성 높이고 싶으면 우변 0을 year 수로 바꾸면 됨 (ex 1910)
            order_value, sorted_order_value = self.aggressive_year(all_orders, obs)
        else:
            order_value = self.evaluate_orders(all_orders, obs)
            order_value = dict(sorted(order_value.items(), key=lambda item: item[0], reverse=True))
            sorted_order_value = dict(sorted(order_value.items(), key=lambda item: item[1], reverse=True))
        sorted_order_value = self._shuffle_with_score(sorted_order_value)
        orders_to_execute = self.get_orders_to_execute(sorted_order_value, obs)
        self.print_sorted = sorted_order_value
        self.print_execute = orders_to_execute
        self.profile_orders(orders_to_execute, obs)

        decided_order = {}
        for recv_order, [sender, idx] in recv_msgs_parsed.items(): # 3. 4/22 HC: sender -> [sender, msg_idx] 로 나올테니 여기서 msg_idx 사용하면 됌
            if recv_order in orders_to_execute:
                decided_order[recv_order] = sender
        if clear_agreed:
            self.agreed_orders.clear()
        self.set_support_request(sorted_order_value, obs)

        execute_order_w_prob = self.get_true_order_prob_msg_idx(order_value, recv_msgs_parsed, obs, orders_to_execute)
        # self.get_true_order_prob_all(order_value, obs)  # 디버깅/데이터 체크 위한 코드
        return execute_order_w_prob, decided_order


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

        order_value = self.evaluate_orders(orders, obs)
        ## value가 높은 순서
        order_value = dict(sorted(order_value.items(), key=lambda item: item[0], reverse=True))
        sorted_order_value = dict(sorted(order_value.items(), key=lambda item: item[1], reverse=True))
        sorted_order_value = self._shuffle_with_score(sorted_order_value)
        for order in sorted_order_value.keys():
            unit_type, unit_loc, order_type, _, dst_infos = self.Order.parse_dipgame(order, obs['loc2power'])

            if order_type == 'RTO':
                if (dst_infos[0] not in done_regions) and (unit_loc not in done_unit_loc):
                    done_regions.append(dst_infos[0])
                    done_unit_loc.append(unit_loc)
                    orders_to_execute.append([order, 0])  # set order true prob ZERO

            elif order_type == 'DSB':
                if unit_loc not in done_regions and (unit_loc not in done_unit_loc):
                    done_regions.append(unit_loc)
                    done_unit_loc.append(unit_loc)
                    orders_to_execute.append([order, 0])  # set order true prob ZERO

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

        sorted_order_value = self._shuffle_with_score(sorted_order_value)
        for order in sorted_order_value.keys():
            i = 0
            while (i < len(orders)) and (len(orders_to_execute) < n_builds):
                i += 1
                unit_type, unit_loc, order_type, _, _ = self.Order.parse_dipgame(order, obs['loc2power'])
                if order_type == "BLD":
                    if unit_loc not in destinations:
                        destinations.append(unit_loc)
                        orders_to_execute.append([order, 0])  # set order true prob ZERO

                elif order_type == 'WAIVE':
                    orders_to_execute.append([order, 0])  # set order true prob ZERO

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
        sorted_order_value = self._shuffle_with_score(sorted_order_value)

        for order in sorted_order_value.keys():
            i = 0
            while (i < len(orders)) and (len(orders_to_execute) < n_removals):
                i += 1
                unit_type, unit_loc, _, _, _ = self.Order.parse_dipgame(order, obs['loc2power'])
                if unit_loc not in done_regions:
                    done_regions.append(unit_loc)
                    orders_to_execute.append([order, 0])  # set order true prob ZERO

        return orders_to_execute

    '''
    evaluate the order
    '''
    def evaluate_orders(self, orders, obs):
        map_dict = dict()
        rela_dict = dict()
        order_dict = dict()
        self.map_value_dict = dict()

        infos = {
            "ratio": self.ratio,
            # 모든 received msg 를 이용해서 평가하도록 함. nego 단계에서는 agreed_order가 발생하기 전이므로.
            "agreed_orders": self.recv_msgs_parsed,  # self.agreed_orders,
            "needs_support": self.need_support,
        }

        for adviser in self.advisers.values():
            adviser.before_phase(obs)

        # for analyzing, later will be removed
        if (obs['name'][-2:] in ['SM', 'RM']) and (obs['name'][-1] == 'M'):
            self.save_infos['map'][obs['name']] = dict()
            self.save_infos['rela'][obs['name']] = dict()

        for name, adviser in self.advisers.items():
            # print('---------------------------', name)
            for order in sorted(orders):
                if name == "map":
                    if not (obs["name"][-1] == "M" and len(obs["name"]) == 6) or obs["name"] != 1901:
                        self.map_value_dict[order] = adviser.evaluation(order, obs, infos)
                    map_eval = self.map_value_dict[order]
                    map_dict[order] = map_eval
                elif name == "relation":
                    rela_eval = adviser.evaluation(order, obs, infos)
                    rela_dict[order] = rela_eval
                elif name == "team":
                    team_eval = adviser.evaluation(order, obs, infos)
                    map_dict[order] *= team_eval
                elif name == "fortune":
                    fortune_eval = adviser.evaluation(order, obs, infos)
                    map_dict[order] *= fortune_eval
            if name == "map":
                values = list(map_dict.values())
                if len(values) == 1:
                    weighted_map_dict = {k: 0.5 * adviser.weight for k, v in map_dict.items()}
                else:
                    # 현재 map_dict 에 있는 값으로 min-max norm
                    weighted_map_dict = {k: (v - min(values)) / (max(values) - min(values) + 1e-5) * adviser.weight for k, v in map_dict.items()}
                print_ori_map_raw = {k: (v - min(values)) / (max(values) - min(values) + 1e-5) for k, v in map_dict.items()}
                self.print_ori_map = dict(sorted(print_ori_map_raw.items(), key=lambda x: x[1], reverse=True))
                if (obs['name'][-2:] in ['SM', 'RM']) and (obs['name'][-1] == 'M'):
                    self.save_infos['map'][obs['name']] = copy.deepcopy(weighted_map_dict) # For analyzing, this will be removed
                    self.save_infos['trust'][obs['name']] = copy.deepcopy(self.ratio) # 3/27 self.effective_ratio -> ratio
                self.print_weighted_map = dict(sorted(weighted_map_dict.items(), key=lambda x: x[1], reverse=True))
                debug = True
            if name == 'relation':
                values = list(rela_dict.values())
                if len(values) == 1:
                    weighted_rela_dict = {k: 0.5 * adviser.weight for k, v in rela_dict.items()}
                else:
                    weighted_rela_dict = {k: (v - min(values)) / (max(values) - min(values) + 1e-5) * adviser.weight for k, v in rela_dict.items()}
                print_ori_map_raw = {k: (v - min(values)) / (max(values) - min(values) + 1e-5) for k, v in rela_dict.items()}
                self.print_ori_rel = dict(sorted(print_ori_map_raw.items(), key=lambda x: x[1], reverse=True))
                if (obs['name'][-2:] not in ['SM', 'RM']) and (obs['name'][-1] == 'M'):
                    self.save_infos['rela'][obs['name']] = copy.deepcopy(weighted_rela_dict) # For analyzing, this will be removed
                self.print_weighted_rel = dict(sorted(weighted_rela_dict.items(), key=lambda x: x[1], reverse=True))
                debug = True

            if name == "map" and obs["name"][-1] == "M" and len(obs["name"]) == 6:
                self.map_value_dict = dict()
        for key in rela_dict.keys():
            order_dict[key] = weighted_rela_dict[key] + weighted_map_dict.get(key, 0)

        self.analyze_order_values(order_dict, obs)

        return order_dict

    def aggressive_year(self, orders, obs):
        order_dict = dict()
        move_order_dict = dict()
        other_order_dict = dict()
        for order in sorted(orders):
            unit_type, unit_loc, order_type, src_infos, dst_infos = self.order.parse_dipgame(order, obs['loc2power'])
            if order_type == 'MTO':
                move_order_dict[order] = 1
                order_dict[order] = 1
            else:
                other_order_dict[order] = 0
                order_dict[order] = 0
        sorted_move_order_dict = self.shuffle_dict(move_order_dict)

        sorted_order_dict = sorted_move_order_dict.copy()
        # dict2의 항목을 merged_dict에 추가
        sorted_order_dict.update(other_order_dict)

        return order_dict, sorted_order_dict

    def shuffle_dict(self, d):
        items = list(d.items())
        random.shuffle(items)
        shuffled_dict = dict(items)
        return shuffled_dict

    def analyze_order_values(self, order_dict, obs):
        onestep_values = {"HLD": [0, 0],
                       "MTO": [0, 0],
                       "SUP": [0, 0],
                       "SUPMTO": [0, 0],
                       "CVY": [0, 0],
                       "CVY_MOVE": [0, 0],
                       "DSB": [0, 0],
                       "RTO": [0, 0],
                       "BLD": [0, 0],
                       "WVE": [0, 0],
                       }


        for order, value in order_dict.items():
            unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(order, obs['loc2power'])
            self.values[order_type][0] += 1
            self.values[order_type][1] += value
            onestep_values[order_type][0] += 1
            onestep_values[order_type][1] += value
        value_dict = {}
        for k, v in onestep_values.items():
            if v[0] == 0:
                value_dict[k] = 0
            else:
                value_dict[k] = round(v[1] / v[0], 4)
        value_dict = dict(reversed(sorted(value_dict.items(), key=lambda x: x[1])))
        # print(self.me, value_dict)

    def check_best_order(self, orders, obs):
        # orders 는 이미 value 로 정렬된 값
        order_by_region = dict()  # 최종 결정된 order를 region으로 묶은 것
        destinations = []  # 최종 결정된 order unit의 도착지
        check_order = {}  # 만약에 내가 도우려고 한 order를 내가 실행하지 않게 된다면, 그 sup order 대신 할 행동
        for order in orders.keys():
            unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(order, obs['loc2power'])
            get_controlled_regions = [unit.split(' ')[1] for unit in obs['units'][self.me]]
            # unit_loc in get_controlled_regions 는 작동 안 하는 조건인 것 같음
            if (unit_loc in get_controlled_regions) and (unit_loc not in order_by_region.keys()):
                final_destination = self.adviser.get_final_destination(order, obs['loc2power'])
                owner = obs['loc2power'][final_destination]
                # 현재 Order가 MTO면 => 공격지가 내 땅이 아니고 & destinations에 도착지가 없고 & 내 unit이나 다른 나라의 unit의 도움을 받을 수 있고
                # & 내 unit의 도움만 가능하지만 그 unit이 order_by_region에 key로 없으면 => order_by_region에 넣음
                # 즉, 나 공격 하는거 빼고 & 더 높은 value를 가진 order로 인해 내 unit이 가기로 한 곳 빼고 & 도움이 필요한 order인데 도와줄 Unit이 없으면 빼고
                # & 도움 필요하고 내 unit만 도울 수 있는데 그 도울 unit이 이미 다른 order 하기로 했으면 뺌
                # CVY_MOVE 도 MTO로 분류되는듯
                if (order_type == 'MTO') and (final_destination not in destinations) and (self.me != owner):
                    # {unit_type} {unit_loc} - {dst}
                    if (order in self.need_support.keys()) and (self.need_support[order] == []) and (
                            self.other_need_support[order] == []):
                        continue
                    if (order in self.need_support.keys()) and (len(self.need_support[order]) > 0) and (
                            self.other_need_support[order] == []):
                        for supporter in self.need_support[order][0].keys():  # TODO 4/21 왜 for문인데 [0]만 도는지 의문
                            if supporter in order_by_region.keys():  # TODO 4/21, supporter가 이미 이 order를 돕기로 할 경우는 안 생기나?
                                continue
                    order_by_region[unit_loc] = order
                    destinations.append(dst_infos[0])

                elif order_type == 'SUP':  # 3/31 todo: 왜 SUPPORT는 여기 조건문에서 빠졌는지 알아내야 함 + 510번째 올바른지 확인
                    whom_sup = src_infos[1]  # 내 sup를 받는 나라
                    if whom_sup == self.me:
                        hold_order = order.split(" ")[-2] + " " + order.split(" ")[-1] + " H"
                        check_order[order.split(" ")[-1]] = [hold_order, unit_loc, order]
                    order_by_region[unit_loc] = order
                    destinations.append(src_infos[0])

                elif order_type == 'HLD':
                    # {unit_type} {unit_loc} H
                    order_by_region[unit_loc] = order
                    destinations.append(unit_loc)

                elif order_type == 'SUPMTO' or order_type == "CVY":
                    # {unit_type} {unit_loc} S {src_type} {src_loc} - {dst}
                    supported_power = obs['loc2power'][src_infos[0]]
                    attacked_power = obs['loc2power'][dst_infos[0]]
                    if supported_power == self.me:
                        move_order = order.split(" ")[-4] + " " + order.split(" ")[-3] + " " + order.split(" ")[
                            -2] + " " + order.split(" ")[-1]
                        check_order[order.split(" ")[-3]] = [move_order, unit_loc, order]
                    if (supported_power != None) and (attacked_power != None):
                        order_by_region[unit_loc] = order
                        destinations.append(dst_infos[0])

        return order_by_region, check_order

    def get_orders_to_execute(self, orders, obs):  # orders : dictionary [order : values]
        supported = []
        order_by_region, check_order = self.check_best_order(orders, obs)

        updated_orders = copy.deepcopy(orders)
        while True:
            check_done = True
            for loc, order in check_order.items():
                if order_by_region[loc] != order[0]:
                    del updated_orders[order[2]]
                    check_done = False
            if check_done:
                break
            order_by_region, check_order = self.check_best_order(updated_orders, obs)

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
                '''
                여기 부분 peace, war -> self.ratio 기반으로 돌아가게 바꿔야 함
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
                '''

            elif order_type == 'HLD':
                self.holds += 1

            elif order_type == 'SUPMTO' or order_type == 'CVY':
                sup_mto_order = order
                self.supports += 1
                power = obs['loc2power'][sup_mto_order.split(' ')[1]]
                if self.me != power:
                    self.supports_ally += 1

    def action_ratio(self, obs, powers, ratio):
        # TODO betray : 이전 phase에서 공격했으면, 이번 phase order 계획 다 세워놓고 이제서야 바꾸는건데, 이것도 현재 order로 바꾸는게 맞는거 아닌지
        other_powers = [power for power in powers if self.me not in power]
        other_prev_orders = {power: obs['prev_orders'][power] for power in other_powers}

        if obs['prev_orders'][self.me] == None:  # 첫 phase
            return ratio

        for attack_power, orders in other_prev_orders.items():
            for order in orders:
                order = order[0]
                unit_type, unit_loc, order_type, src_infos, dst_infos = self.Order.parse_dipgame(order, obs['loc2power'])
                if (order_type == 'MTO') or (order_type == 'CVY_MOVE') or (order_type == 'SUPMTO'):
                    if dst_infos[1] and (dst_infos[1] == self.me):
                        ratio[attack_power] -= 0.005
            if ratio[attack_power] < -1:
                ratio[attack_power] = -1

        return ratio

    def agreed_order_ratio(self, executed_orders, ratio):
        # TODO betray : self.promised_orders와 order의 시기가 일치하는지 확인 필요. 이거 아니면 전체 데이터 다시 모아야 맞는거임
        executed_order = {power: [o[0] for o in order] for power, order in executed_orders.items()}
        self.betrayal = {power: [] for power in executed_order.keys()}
        self.fulfill = {power: [] for power in executed_order.keys()}
        # power_promised가 해주기로 약속했던 order들 전체 집합이 promised_order_list
        for power_promised, promised_order_list in self.promised_orders.items():
            if len(promised_order_list) > 0:  # 약속한 order가 있으면
                # 실제로 약속을 이행하지 않은 order의 개수
                executed_order_list = executed_order[power_promised]
                betray_order_list = set(promised_order_list) - set(executed_order_list)
                if len(betray_order_list) != 0:
                    self.betrayal[power_promised] += list(betray_order_list)
                    ratio[power_promised] -= 0.01  # TODO betray : 배신하면 깎는거 맞는지, 개수 상관없이로 의도한 것이 맞는지 체크

                # 실제로 약속을 이행한 order의 개수
                fulfill_order_list = set(promised_order_list) & set(executed_order_list)
                if len(fulfill_order_list) != 0:
                    self.fulfill[power_promised] += list(fulfill_order_list)

            if ratio[power_promised] < -1:
                ratio[power_promised] = -1

        return ratio