# 2. 일반 디플로매시 시작하기
import sys, os
import copy
import numpy as np
import gym
from tqdm import tqdm
sys.path.append(os.getcwd())
from gym.envs.registration import register

from dipbluebot.dipblue import DipBlue
from dipbluebot.dipblue_order_handler import DipBlueOrderHandler
from dipbluebot.adviser.map_tactician import AdviserMapTactician
from dipbluebot.adviser.relation_controller import AdviserRelationController
from utils.utils import *
register(
    id='PressDiplomacyEnv-v0',
    entry_point='environment.env:PressDiplomacyEnv')

class Storage(object):

    def __init__(self, env, env_type, weights, num_episode, num_same_population_episode):
        self.env = env
        self.weights = np.array(weights)
        self.num_population = len(weights)
        self.num_episode = num_episode
        self.num_sub_episode = num_same_population_episode

        if env_type == 1:  # 7 players
            self.num_powers = 7
            self.obss_shape = (81, 75)
        elif env_type == 2:  # samll env
            self.num_powers = 3
            self.obss_shape = (28, 47)

    def extract_epi(self, sampled_weights):
        # Match weights to each powers randomly
        epi_log = {  # SAVE
            "infos": [],
            "internal_states": [],
            "orders": [],
            "messages": [],
            "static_infos": {}
        }

        selfish_epi_log = {  # SAVE
            "orders": [],
        }

        naive_epi_log = {
            "orders": [],
        }
        obss, infos = self.env.reset()
        epi_log['static_infos'] = copy.deepcopy(self.env.static_infos)
        epi_log['static_infos'].pop('powers')
        done = False
        agents = {power: DipBlue(self.env.static_infos, power, weights=weight) for power, weight in zip(list(self.env.static_infos['powers']), sampled_weights)}
        print(sampled_weights)
        while not done:

            # selfish_agents = copy.deepcopy(agents)
            # naive_agents = copy.deepcopy(agents)
            # for power in self.env.static_infos['powers']:
            #     selfish_agents[power].dipblue_handler = DipBlueOrderHandler(agents[power].static_dict, agents[power].me, [0.99, 0.01])
            #     selfish_agents[power].negotiator = agents[power].negotiator
            #     naive_agents[power].dipblue_handler = DipBlueOrderHandler(agents[power].static_dict, agents[power].me, [0.01, 0.99])

            start_phase = self.env.game.phase.split(" ")[0] + self.env.game.phase.split(" ")[-1]
            orders = {power: [] for power in list(self.env.static_infos["powers"])}
            messages = {power: [] for power in list(self.env.static_infos["powers"])}
        
            selfish_orders = copy.deepcopy(orders)
            selfish_messages = copy.deepcopy(messages)

            naive_orders = copy.deepcopy(orders)
            naive_messages = copy.deepcopy(messages)
            # e.g. S1951SM (STATE) -> S1951OM (Message) -> next (Order)
            # e.g. F1951SM (STATE) -> F1951OM (Message) -> next (Order)

            while self.env.game.phase.split(" ")[0]  + self.env.game.phase.split(" ")[-1]== start_phase:
                if self.env.game.is_game_done:
                    done = True
                    break

                org_weight = {}
                if not self.env.is_nego:  # action phase
                    for power, agent in agents.items():
                        org_weight[power] = agent.dipblue_handler.weight

                    for power, agent in agents.items():
                        agreed_orders = copy.deepcopy(agent.agreed_orders)
                        power_orders, _, prev_order_clear = agent.act(infos)
                        self.env.submit((power, power_orders), prev_order_clear)
                        orders[power] = power_orders
                        org_weight[power] = agent.dipblue_handler.weight
                    #for power, sagent in agents.items():

                        agent.dipblue_handler.weight = [1, 0]
                        agent.dipblue_handler.advisers = {"map": AdviserMapTactician(agent.static_dict, agent.me, agent.dipblue_handler.weight[0]),
                                                           "relation": AdviserRelationController(agent.static_dict, agent.me, agent.dipblue_handler.weight[1])}
                        agent.agreed_orders = agreed_orders
                        power_orders, _, _ = agent.act(infos)
                        selfish_orders[power] = power_orders

                    #for power, nagent in agents.items():
                        agent.dipblue_handler.weight = [0, 1]
                        agent.dipblue_handler.advisers = {"map": AdviserMapTactician(agent.static_dict, agent.me, agent.dipblue_handler.weight[0]),
                                                           "relation": AdviserRelationController(agent.static_dict, agent.me, agent.dipblue_handler.weight[1])}
                        agent.agreed_orders = agreed_orders
                        power_orders, _, _ = agent.act(infos)
                        naive_orders[power] = power_orders

                    for power, agent in agents.items():
                        agent.dipblue_handler.weight = org_weight[power]
                        agent.dipblue_handler.advisers =  {"map": AdviserMapTactician(agent.static_dict, agent.me, org_weight[power][0]),
                                                         "relation": AdviserRelationController(agent.static_dict, agent.me, org_weight[power][1])}

                else:  # nego phase
                    for power, agent in agents.items():
                        negos, agreed_orders, _ = agent.act(infos)
                        self.env.submit(negos, agreed_orders)

                board_state = copy.deepcopy(infos)
                obs, rew, done, infos = self.env.step(None)
                for power in list(self.env.static_infos["powers"]):
                    messages[power] += infos['messages'][power]

            if 'messages' in board_state.keys():
                board_state.pop('messages')
            epi_log["orders"].append(copy.deepcopy(orders))
            epi_log["infos"].append(copy.deepcopy(board_state))
            epi_log["messages"].append(copy.deepcopy(messages))
            epi_log["internal_states"].append(copy.deepcopy({power: [agent.peace, agent.war, agent.trust, agent.ratio]
                                               for power, agent in agents.items()}))
            selfish_epi_log["orders"].append(copy.deepcopy(selfish_orders))
            naive_epi_log["orders"].append(copy.deepcopy(naive_orders))
            if done:
                break

        return epi_log, selfish_epi_log, naive_epi_log

if __name__ == "__main__":
    population_pool = load_weights("preprocessing/population_pool_30.pickle")
    env = gym.make('PressDiplomacyEnv-v0')
    env.game_type = "standard"

    np.random.randint(population_pool.shape[0], size=7)
    shuffle = np.random.randint(population_pool.shape[0], size=7)

    COUNTRIES = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    same_orders_counts = {k:0 for k in COUNTRIES}

    total_step = 0
    for i in tqdm(range(300)):
        storage = Storage(env, 1, population_pool[shuffle], 1, 5)
        a,b,c = storage.extract_epi(population_pool[shuffle])
        for step in range(len(b['orders'])):
            for k in COUNTRIES:
                if b['orders'][step][k] == c['orders'][step][k]:
                    same_orders_counts[k] +=1
            total_step+=1
    print(same_orders_counts, total_step)
        



