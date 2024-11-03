import sys, os
import copy
import numpy as np
import yaml
sys.path.append(os.getcwd())
from dipbluebot.dipblue import DipBlue


class Storage(object):

    def __init__(self, env, env_type, weights, num_episode, num_same_population_episode):
        self.env = env
        self.weights = np.array(weights)
        self.num_population = len(weights)
        self.num_episode = num_episode
        self.num_sub_episode = num_same_population_episode
        config_path = "./configs/{}.yaml".format(env_type)
        with open(config_path) as f:
            self.yd = yaml.load(f, Loader=yaml.FullLoader)
        self.num_powers = self.yd["num_agent"]
        # self.obss_shape = (self.yd["num_loc"], self.yd[""])  # 75, 47
        # self.env.reset()

    def extract_epi(self, sampled_weights):
        # Match weights to each powers randomly
        epi_log = {  # SAVE
            "infos": [],
            "internal_states": [],
            "orders": [],
            "messages": [],
            "static_infos": {},
            "betrays": [],
            "fulfills": []
        }

        self.env.reset()
        obss, infos = self.env.reset()
        epi_log['static_infos'] = copy.deepcopy(self.env.static_infos)
        epi_log['static_infos'].pop('powers')
        done = False
        # print(self.env.static_infos['powers'], sampled_weights)
        agents = {power: DipBlue(self.env.static_infos, power, weights=weight) for power, weight in zip(list(self.env.static_infos['powers']), sampled_weights)}
        messages_len = {power: {power: 0 for power in list(self.env.static_infos["powers"])} for power in
                        list(self.env.static_infos["powers"])}
        while self.env.is_done() is False:
            start_phase = self.env.game.phase.split(" ")[0] + self.env.game.phase.split(" ")[-1]
            orders = {power: [] for power in list(self.env.static_infos["powers"])}
            messages = {power: [] for power in list(self.env.static_infos["powers"])}
            betrays = {power: [] for power in list(self.env.static_infos["powers"])}
            fulfills = {power: [] for power in list(self.env.static_infos["powers"])}
            # e.g. S1951SM (STATE) -> S1951OM (Message) -> next (Order)
            # e.g. F1951SM (STATE) -> F1951OM (Message) -> next (Order)

            while self.env.game.phase.split(" ")[0] + self.env.game.phase.split(" ")[-1] == start_phase:

                if not self.env.is_nego:  # action phase
                    for power, agent in agents.items():
                        power_orders, _, prev_order_clear = agent.act(infos)
                        only_orders = [order_tuple[0] for order_tuple in power_orders]
                        self.env.submit((power, power_orders), prev_order_clear)
                        orders[power] = power_orders
                        # print(power, power_orders)

                    ratio_dict = {power: [agent.trust, agent.ratio] for power, agent in agents.items()}

                    # TODO betray : 여기서 ratio 를 깎아도 데이터 모았을 때 문제 없는지 체크. 지금 막 한 order의 ratio를 깎았음! 그래서 ratio를 위에서 미리 저장
                    for power, agent in agents.items():
                        agent.after_act(orders)
                        betrays[power] = agent.betrayal
                        fulfills[power] = agent.fulfill

                else:  # nego phase
                    for power, agent in agents.items():
                        negos, agreed_orders, _ = agent.act(infos)
                        self.env.submit(negos, agreed_orders)
                        # print("negos : ", negos)
                        # print("agreed_orders : ", agreed_orders)
                    ratio_dict = {power: [agent.trust, agent.ratio] for power, agent in agents.items()}
                # print('--------------------------------')
                board_state = copy.deepcopy(infos)
                obs, rew, done, infos = self.env.step(None)
                for me_power in list(self.env.static_infos["powers"]):
                    messages[me_power] += infos['messages'][me_power]
                    for msg in infos['messages'][me_power]:
                        if 'PROPOSE' in msg[0]['message']:
                            messages_len[me_power][msg[0]['sender']] += 1 # IF OR exists, count
                        #     if me_power == 'RUSSIA':
                        #         print(board_state['name'], msg[0]['message'])
                        # else:
                        #     if msg[0]['sender'] == 'RUSSIA' and 'REJECT' in msg[0]['message']:
                        #         print(board_state['name'], msg[0]['message'])
            if 'messages' in board_state.keys():
                board_state.pop('messages')
            epi_log["orders"].append(copy.deepcopy(orders))
            epi_log["infos"].append(copy.deepcopy(board_state))
            epi_log["messages"].append(copy.deepcopy(messages))
            epi_log["betrays"].append(copy.deepcopy(betrays))  # 첫번째 key가 배신 당한 나라. 그 안에 있는 key가 배신 한 사람
            epi_log["fulfills"].append(copy.deepcopy(fulfills))

            epi_log["internal_states"].append(copy.deepcopy(ratio_dict))
            if done:
                epi_log["num_messages_epi"] = copy.deepcopy(messages_len)
                # print(epi_log['messages'])
                # print(epi_log["orders"])
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