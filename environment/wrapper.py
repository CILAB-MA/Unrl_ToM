from gym import Wrapper
from diplomacy import Game
import os, sys
sys.path.append(os.getcwd())
import copy
import numpy as np
import gym

from dipbluebot.dipblue import DipBlue
from environment.get_state import Np_state
from environment.l1_negotiation import Negotiation


class PressDiplomacyWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.env.game_type = "small"
        self.env.reset()
        self.static_infos = self.env.static_infos
        self.num_powers = len(self.env.powers)
        self.num_units = 3  # A, F, None
        self.num_order = 4  # H, -, S, C
        self.num_locs = len(self.env.static_infos["locs"])
        self.np_state = Np_state(self.num_powers, self.static_infos)

    @property
    def game(self):
        return self.env.game

    @property
    def current_year(self):
        return self.env.current_year

    @property
    def game_id(self):
        if self.game:
            return self.game.game_id
        return ''

    @property
    def is_done(self):
        return self.env.is_done()

    def process(self, **kwargs):
        return self.env.process(**kwargs)

    def step(self):
        done = self._process_one_phase()
        if done:
            return np.zeros((81, 75)), [np.zeros((4, 76)), np.zeros((4,)), np.zeros((4,)), np.zeros((4,))], True
        phase_history = self.env.game.get_phase_history(from_phase=-6)
        phase_history = [phase.to_dict() for phase in phase_history]
        phase_history.reverse()
        last_phase_name = [phase['name'][-1] for phase in phase_history]
        phase_name = [phase['name'] for phase in phase_history]
        prev_order_ind = last_phase_name.index('M')
        phase_history = phase_history[prev_order_ind + 1:]
        last_phase_name = last_phase_name[prev_order_ind + 1: ]
        phase_name = phase_name[prev_order_ind + 1:]
        prev_order_ind = last_phase_name.index('M')
        prev_order_obss = phase_history[prev_order_ind]
        all_msgs, send_padding, masking, sender_response = self.np_state.message_state_maker(self.messages_saved, self.me, self.other)

        msgs = (all_msgs, send_padding, masking, sender_response)
        prev_order_state = self.np_state.prev_order_state_maker(prev_order_obss)
        board_state = self.np_state.prev_board_state_maker(self.prev_info)
        obss_np = np.concatenate([board_state, prev_order_state], axis=-1)
        self.messages_saved = []

        return obss_np, msgs, done

    def reset(self, weights, me, other):
        self.me = me
        self.other = other
        self.obss, self.infos = self.env.reset()
        self.time = 0
        self.agents = {power: DipBlue(self.static_infos, power, weights=weight) for power, weight in
                    zip(list(self.static_infos['powers']), weights)}

        _ = self._process_one_phase()

        board_state = self.np_state.prev_board_state_maker(self.prev_info)
        prev_order_state = np.zeros((self.num_locs, self.num_units + 4 * self.num_powers + self.num_order + 5))
        obss_np = np.concatenate([board_state, prev_order_state], axis=-1)

        all_msgs, send_padding, masking, sender_response = self.np_state.message_state_maker(self.messages_saved, self.me, self.other)
        # for first observation for tom experiment

        return obss_np, (all_msgs, send_padding, masking, sender_response)

    def _process_one_phase(self):
        self.prev_obss = copy.deepcopy(self.obss)
        self.prev_info = copy.deepcopy(self.infos)
        start_phase = self.env.game.phase.split(" ")[0]
        while self.env.game.phase.split(" ")[0] == start_phase or self.env.game.phase.split(" ")[0] == "WINTER":
            if self.game.is_game_done:
                done = True
                break
            if not self.env.is_nego:  # action phase
                for power, agent in self.agents.items():
                    power_orders, _, prev_order_clear = agent.act(self.infos)
                    self.env.submit((power, power_orders), prev_order_clear)
            else:  # nego phase
                for power, agent in self.agents.items():
                    negos, agreed_orders, _ = agent.act(self.infos)
                    self.env.submit(negos, agreed_orders)

            self.obs, rew, done, self.infos = self.env.step(None)

            if self.env.game.phase.split(" ")[0] == start_phase and self.env.game.phase.split(" ")[0] != "WINTER":
                self.messages_saved = copy.deepcopy(self.infos["messages"])
            if done:
                break

        return done


class AssignAgent(PressDiplomacyWrapper):

    def __init__(self, env, players, power_assignments):

        super(AssignAgent, self).__init__(env)

        game = self.game or Game()
        self._powers = power_assignments
        self._players = players

    def reset(self, **kwargs):
        return_var = self.env.reset(**kwargs)
        self.game.note = ' / '.join([power_name[:3] for power_name in self._powers])
        return return_var


def test_dummy_players():
    import random
    weight1 = [0.5, 0, -0.2]
    weights = [weight1, weight1, weight1, weight1, weight1, weight1, weight1]
    env = gym.make('PressDiplomacyEnv-v0')
    powers = ['FRANCE', 'AUSTRIA', 'GERMANY', 'RUSSIA', 'ITALY', 'TURKEY', 'ENGLAND']
    env.reset()
    players = {p:DipBlue(env.static_infos, me=p, weights=w) for p, w in zip(powers, weights)}
    obs, infos = env.reset()

    print(env.is_done())
    print(infos['name'], infos['messages'])
    while env.is_done() is False:
        valid_actions = env.game.get_all_possible_orders()
        if not env.is_nego:
            for power in powers:
                power_orders = [random.choice(valid_actions[loc]) for loc in
                                env.game.get_orderable_locations(power)]
                env.submit((power, power_orders))
            obs, rew, dones, infos = env.step(None)
        else:
            for _ in powers:
                msgs = ['S1901M PROPOSE FRANCE GERMANY AGREE FRANCE PEACE FRANCE'] * 2
                env.submit(msgs)
            obs, rew, done, infos = env.step(None)
            print(infos['name'], infos['messages'])


def make_one_weight(num_agent, feature_dim, others_constant=1.0):
    props = np.full((num_agent, 4), others_constant)
    feat = np.random.normal(0, 1, num_agent)
    props[:, feature_dim] = feat
    return props


def generate_weights(num_agents, num_adviser=4):
    weights = np.random.rand(num_agents, num_adviser)

    return weights * 2 - 1


def test_assign_players(game_type):
    env = gym.make('PressDiplomacyEnv-v0')
    env.game_type = game_type
    if game_type == "standard":
        powers = ['FRANCE', 'AUSTRIA', 'GERMANY', 'RUSSIA', 'ITALY', 'TURKEY', 'ENGLAND']
        weights = generate_weights(7)
    elif game_type == "small":
        powers = ['FRANCE', 'GERMANY', 'ITALY']
        weights = generate_weights(3)
    env.reset()
    players = {p:DipBlue(env.static_infos, me=p, weights=w) for p, w in zip(powers, weights)}
    obs, infos = env.reset()
    import time
    start =time.time()
    count_step=0
    parser = Negotiation()
    while env.is_done() is False:
        count_step += 1
        # print("-------------------------------------------------------------------------------------------------------------------")
        #print(infos['name'])

        # print("-------------------------------------------------------------------------------------------------------------------")
        if not env.is_nego:
            #start = time.time()
            for power in powers:
                power_orders, _, prev_order_clear = players[power].act(infos)
                #power_orders = [random.choice(valid_actions[loc]) for loc in
                #                env.game.get_orderable_locations(power)]
                env.submit((power, power_orders), prev_order_clear)
                # print(power, power_orders)
                if infos['name'][-1] == 'M':
                    print(len(power_orders), infos['name'])
            obs, rew, dones, infos = env.step(None)
            #print('Order Iteration :',time.time() - start)
        else:
            #start = time.time()
            for power in powers:
                negos, agreed_orders, _ = players[power].act(infos)
                env.submit(negos, agreed_orders)
                # print("power : ", power)
                # print_nego=[]
                # for i in range(len(negos)):
                #     if infos['name'][-2:] == "RM":
                #         msg = negos[i][0]
                #     else:
                #         msg = negos[i]
                #     if parser.parse(msg)[3][0] != "ALLIANCE":
                #         print_nego.append(negos[i])
                # print("negos : ", print_nego)
                # print("agreed_orders : ", agreed_orders)
            obs, rew, done, infos = env.step(None)
            #print('Nego Iteration :',time.time() - start)
        # if env.game.render(incl_orders=True) != None:
        #     with open('./output/{}_{}.svg'.format(count_step, infos['name']), 'w') as f:
        #         f.write(env.game.render(incl_orders=True))
    print('One Game  Iteration :', time.time() - start)


if __name__ == '__main__':
    #test_dummy_players()
    from gym.envs.registration import register
    print(sys.path)

    register(
        id='PressDiplomacyEnv-v0',
        entry_point='environment.env:PressDiplomacyEnv')

    game_type = "standard"  # standard, small
    test_assign_players(game_type)

