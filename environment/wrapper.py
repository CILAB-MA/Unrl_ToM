from gym import Wrapper
from diplomacy import Game
import os, sys
sys.path.append(os.getcwd())
sys.path.append("/app/Dip_ToM/")
import copy
import numpy as np
import pandas as pd
import gym
import yaml
import argparse
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
        return copy.deepcopy(self.env.current_year)

    @property
    def game_id(self):
        if self.game:
            return self.game.game_id
        return ''

    @property
    def is_done(self):
        start_year = self.start_year
        curr_year = self.current_year()
        if curr_year - start_year > self.env.num_step:
            self.env.game.draw()
        return self.env.is_done()

    def set_game_type(self, game_type, num_step):
        self.env.game_type = game_type
        self.num_step = num_step

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
        self.agents = {power: DipBlue(self.static_infos, power, weights=weight) for power, weight in
                    zip(list(self.static_infos['powers']), weights)}
        self.start_year = self.current_year()
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


def generate_weights(num_agents, num_adviser=2):
    map_weights = np.random.rand(num_agents, 1)
    relation_weights = 1 - map_weights
    weights = np.column_stack([map_weights, relation_weights])

    return weights


def test_assign_players():
    env = gym.make('PressDiplomacyEnv-v0')

    powers = env.get_powers()
    weights = generate_weights(env.get_num_agent())
    env.reset()
    players = {p: DipBlue(env.static_infos, me=p, weights=w) for p, w in zip(powers, weights)}
    obs, infos = env.reset()
    import time
    start =time.time()
    count_step=0
    parser = Negotiation()
    prev_orders = {p:[] for p in powers}
    num_same_order = 0
    start_year = int(infos['name'][1:5])
    info_list = []
    while env.is_done() is False:
        count_step += 1
        # print("-------------------------------------------------------------------------------------------------------------------")
        # print(infos['name'])
        # print("-------------------------------------------------------------------------------------------------------------------")
        if num_same_order > len(powers) * args.repeat_num:
            break
        # print('SAME ORDER', num_same_order)
        if not env.is_nego:
            # start = time.time()
            # print("-------------------------------------------------------------------------------------------------------------------")
            for power in powers:
                power_orders, _, prev_order_clear = players[power].act(infos)
                if (infos['name'][-1] == 'M') and (infos['name'][0] == 'F'):
                    # print(infos['name'], power, prev_orders[power], power_orders)
                    if power_orders in prev_orders[power]:
                        num_same_order += 1
                    else:
                        num_same_order = 0
                        prev_orders[power].append(power_orders)
                        prev_orders[power] = prev_orders[power][-args.pattern_num:]
                # power_orders = [random.choice(valid_actions[loc]) for loc in
                #                env.game.get_orderable_locations(power)]
                env.submit((power, power_orders), prev_order_clear)

                # if infos['name'][-1] == 'M':
                #     print(len(power_orders), infos['name'])
            obs, rew, dones, infos = env.step(None)
            info_list.append(infos['name'])
            # print('Order Iteration :',time.time() - start)
        else:
            # start = time.time()
            # print("negos")
            # print("-------------------------------------------------------------------------------------------------------------------")
            for power in powers:
                negos, agreed_orders, _ = players[power].act(infos)
                env.submit(negos, agreed_orders)
                # print("power : ", power)
                print_nego=[]
                for i in range(len(negos)):
                    if infos['name'][-2:] == "RM":
                        msg = negos[i][0]
                    else:
                        msg = negos[i]
                    if parser.parse(msg)[3][0] != "ALLIANCE":
                        print_nego.append(negos[i])
                # print("negos : ", power, print_nego)
                # print("agreed_orders : ", agreed_orders)
            obs, rew, done, infos = env.step(None)
            info_list.append(infos['name'])
            # print('Nego Iteration :',time.time() - start)
        # if env.game.render(incl_orders=True) != None:
        #     with open('output/{}_{}.svg'.format(count_step, infos['name']), 'w') as f:
        #         f.write(env.game.render(incl_orders=True))
    if infos['name'] == "COMPLETED":
        end_year = int(str(info_list[-2:-1])[3:7])
    else:
        end_year = int(infos['name'][1:5]) + 1
    print('One Game  Iteration :', time.time() - start)
    print('year_length', end_year - start_year)
    print('end_year', end_year)
    print('start_year', start_year)
    return int(end_year - start_year)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # standard, small, ancmed, pure, clonial, empire, known_world_901, modern, world
    parser.add_argument('--game_type', '-gt', type=str, default="standard")
    parser.add_argument('--test_num', '-tn', type=int, default=100)
    parser.add_argument('--repeat_num', '-rn', type=int, default=3)
    parser.add_argument('--pattern_num', '-pn', type=int, default=3)
    parser.add_argument("--out_path", "-op", type=str, default="./output/")
    args = parser.parse_args()

    #test_dummy_players()
    from gym.envs.registration import register
    config_path = "./configs/{}.yaml".format(args.game_type)
    print(sys.path)
    register(
        id='PressDiplomacyEnv-v0',
        entry_point='environment.env:PressDiplomacyEnv',
        kwargs={'config_path':config_path})

    game_type = args.game_type
    year_length = []
    test_num = args.test_num
    for _ in range(test_num):
        year_length.append(test_assign_players())

    print("game_type :", game_type)
    print("total_year: ", year_length)
    print("game_avg_year : {}, game_std_year : {}, game_max_year : {}".format(np.mean(year_length), np.std(year_length), np.max(year_length)))
    print("final_game_year : {} ".format( np.mean(year_length) + np.std(year_length)) )
    year_length_pd = pd.DataFrame(year_length)
    if not os.path.exists("./output"):
        os.makedirs("./output")
    year_length_pd.to_csv("./output/{}_years_info.csv".format(game_type))


