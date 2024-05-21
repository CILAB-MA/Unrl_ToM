import sys, os
import copy
import numpy as np
from tqdm import tqdm
sys.path.append(os.getcwd())
from dipbluebot.dipblue import DipBlue


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


        obss, infos = self.env.reset()
        epi_log['static_infos'] = copy.deepcopy(self.env.static_infos)
        epi_log['static_infos'].pop('powers')
        done = False
        agents = {power: DipBlue(self.env.static_infos, power, weights=weight) for power, weight in zip(list(self.env.static_infos['powers']), sampled_weights)}

        while not done:
            start_phase = self.env.game.phase.split(" ")[0] + self.env.game.phase.split(" ")[-1]
            orders = {power: [] for power in list(self.env.static_infos["powers"])}
            messages = {power: [] for power in list(self.env.static_infos["powers"])}
            # e.g. S1951SM (STATE) -> S1951OM (Message) -> next (Order)
            # e.g. F1951SM (STATE) -> F1951OM (Message) -> next (Order)

            while self.env.game.phase.split(" ")[0]  + self.env.game.phase.split(" ")[-1]== start_phase:
                if self.env.game.is_game_done:
                    done = True
                    break

                if not self.env.is_nego:  # action phase
                    for power, agent in agents.items():
                        power_orders, _, prev_order_clear = agent.act(infos)
                        self.env.submit((power, power_orders), prev_order_clear)
                        orders[power] = power_orders

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

            if done:
                break

        return epi_log

    def reset(self):
        self.past_trajectories = np.zeros(self.past_trajectories.shape)
        self.current_state = np.zeros(self.current_state.shape)
        self.target_action = np.zeros(self.target_action.shape)
        self.dones = np.zeros(self.dones.shape)
        self.action_count = np.zeros(self.action_count.shape)

    def get_most_act(self):
        action_count = copy.deepcopy(self.action_count)
        action_count /= np.reshape(np.sum(action_count, axis=-1), (-1, 1))

        return np.argmax(action_count, axis=-1), np.max(action_count, axis=-1)