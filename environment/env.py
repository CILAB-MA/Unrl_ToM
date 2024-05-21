
# This is new env after ICML workshop for optimizing

import gym, logging, copy, sys, os
from gym.utils import seeding
from diplomacy import Game, Map, Power
from diplomacy.engine.message import Message
from environment.l1_negotiation import Negotiation
import numpy as np
import torch as tr
from environment.utils import power2loc, loc2power, loc2unit_type
import time

class PressDiplomacyEnv(gym.Env):
    def __init__(self):
        self.game = None
        self.curr_seed = 0
        self.modified_phase = ['S', 'R', 'O']
        self.tmp_commit = []
        self.msg_parser = Negotiation()
        self.prev_order_clear = False

    def set_game_type(self, game_type):
        self.game_type = game_type

    def current_year(self):
        curr_phase = self.game.get_current_phase()
        if curr_phase == 'COMPLETED':
            curr_phase = self._last_known_phase
        return int(curr_phase[1:5]) - self.game.map.first_year + 1

    def game_id(self):
        if self.game:
            return self.game.game_id
        return ''

    def is_done(self):
        return self.game.is_game_done

    def _change_nego(self, name):
        if name[-2] not in ['S', 'R']:
            self.is_nego = False
        else:
            self.is_nego = True

    def process(self):
        self.game.process()
        self.state = self.game.get_state()
        infos = copy.deepcopy(self.state)
        if self.state['name'][-1] == 'M':
            infos['name'] = self.state['name'][:-1] +'SM'
        else:
            infos['name'] = self.state['name']
        # self.game.powers
        infos['eliminated'] = {power_name: power.is_eliminated() for power_name, power in self.game.powers.items()}
        infos['loc2power'] = loc2power(self.state, self.static_infos['locs'])
        infos['loc2unit_type'] = loc2unit_type(self.state, self.static_infos['all_units'])
        infos['power2loc'] = power2loc(self.state)
        if self.prev_order_clear:
            self.prev_orders = {k: None for k in self.powers}
        infos['prev_orders'] = copy.deepcopy(self.prev_orders)
        infos['prev_agreed'] = None
        infos['valid_actions'] = self.game.get_all_possible_orders()
        infos['messages'] = {p: [] for p in self.powers}

        self.phs_ind = -1
        self.msg_ind = 0
        self.prev_phase = infos['name']
        self.prev_infos = None
        self.message_pair = []
        self.previous_agreed = {k: [] for k in self.powers}
        self.infos = copy.deepcopy(infos)
        self._change_nego(infos['name'])
        return infos

    def nego_process(self):
        infos = copy.deepcopy(self.infos)
        infos['messages'] = {p:[] for p in self.powers}
        infos['messages'] = copy.deepcopy(self.tmp_messages)
        self.tmp_messages = {p: [] for p in self.powers}
        # update phase name
        if infos['name'][-2] == 'S':
            infos['name'] = infos['name'][:-2] + 'RM'
        elif infos['name'][-2] == 'R':
            infos['name'] = infos['name'][:-2] + 'M'
        self.prev_phase = infos['name']
        self.infos = copy.deepcopy(infos)
        self._change_nego(infos['name'])
        return infos

    def seed(self, seed=0):
        np.random.seed(seed)
        if self.use_torch:
            tr.random.seed(seed)

    def step(self, dummy):
        obs, rew, done = None, None, None #TODO
        if self.is_nego:
            infos = self.nego_process()
        else:
            infos = self.process()
        return obs, rew, done, infos

    def submit(self, actions, other_infos):
        # other infos are agreed_orders and prev_orders_clear
        if self.is_nego:
            self.nego_step(actions, other_infos)
        else:
            self.order_step(actions, other_infos)

    def order_step(self, agent_orders, prev_order_clear):
        power_name, orders = agent_orders
        if self.game.get_current_phase()[-1] == 'R':
            orders = [order.replace(' - ', 'R') for order in orders]
        orders = [order for order in orders if order != 'WAIVE']
        self.game.set_orders(power_name, orders, expand=False)
        self.prev_orders[power_name] = orders # if we can get order in get_state(), this will be removed.

        self.prev_order_clear = prev_order_clear

    def nego_step(self, messages, agreed_orders):
        for message in messages:
            if self.infos['name'][-2] == 'R':
                cont, msg_ind = message
                _, _, _, _, sender, receiver, cont = self.msg_parser.parse(cont)
                cont = dict(sender=sender, message=cont)
                self.tmp_messages[receiver].append((cont, msg_ind + 1))
            else:
                _, _, _, _, sender, receiver, message = self.msg_parser.parse(message)
                message = dict(sender=sender, message=message)
                self.tmp_messages[receiver].append((message, self.msg_ind))
                self.msg_ind += 2
            #self.game.add_message(Message(sender=sender,
            #                              recipient=receiver,
            #                              message=message,
            #                              phase=self.game.get_current_phase()))
    def _make_static_infos(self):
        locs = [l.upper() for l in self.map.locs if self.map.area_type(l) not in ['SHUT', None]]
        locs_lower = [l for l in self.map.locs if self.map.area_type(l) not in ['SHUT', None]]
        scs = copy.deepcopy(self.map.scs)
        adjs = np.zeros((len(locs), len(locs)), dtype=np.bool)
        loc_keys = self.map.loc_abut
        area_type = {l.upper():self.map.loc_type[l] for l in locs_lower}
        loc_abuts = {k.upper(): [v.upper() for v in vals if v in locs_lower] for k, vals in loc_keys.items() if k in locs_lower}
        self.static_infos = dict(locs=locs, loc_abut=loc_abuts,
                                scs=scs, area_type=area_type,
                                all_units=['{} {}'.format(unit_type, loc.upper())
                                            for unit_type in 'AF' for loc in locs_lower
                                            if self.map.is_valid_unit('{} {}'.format(unit_type, loc.upper()))],
                                powers=self.powers,
                                abut_list={loc.upper(): [abut.upper() for abut in
                                                          self.map.abut_list(loc.upper(), incl_no_coast=True) if abut in locs_lower]
                                            for loc in locs_lower})

    def reset(self):
        #if 'environment' not in os.getcwd():
        #    sys.path.append(os.getcwd()+'/environment')
        if self.game_type == "standard":
            self.game = Game(meta_rules=['PRESS'])
        elif self.game_type == "small":
            if 'environment' not in os.getcwd():
                self.game = Game(meta_rules=['PRESS'], map_name="./environment/small.map")
            else:
                self.game = Game(meta_rules=['PRESS'], map_name="small.map")
        self.powers = self.game.powers
        self.map = Map(self.game.map.name)
        self._make_static_infos()
        self._last_known_phase = self.game.get_current_phase()
        self.is_nego = True

        # Call States...
        obs = None #TODO
        self.state = self.game.get_state()
        infos = copy.deepcopy(self.state)
        infos['name'] = self.state['name'][:-1] + 'SM'
        infos['eliminated'] = {power_name:power.is_eliminated() for power_name, power in self.game.powers.items()}
        infos['loc2power'] = loc2power(self.state, self.static_infos['locs'])
        infos['loc2unit_type'] = loc2unit_type(self.state, self.static_infos['all_units'])
        infos['power2loc'] = power2loc(self.state)
        infos['valid_actions'] = self.game.get_all_possible_orders()
        infos['messages'] = {p:[] for p in self.powers}
        infos['prev_agreed'] = {k:None for k in self.powers}
        self.prev_orders = {k:None for k in self.powers}
        infos['prev_orders'] = copy.deepcopy(self.prev_orders)
        self.tmp_messages = {p: [] for p in self.powers}
        self.phs_ind = -1
        self.msg_ind = 0
        self.prev_phase = infos['name']
        self.prev_infos = None
        self.message_pair = {}
        self.infos = copy.deepcopy(infos)
        return obs, infos,

    def get_powers(self):
        return self.game.powers.keys()

    def get_saved_game(self):
        pass
