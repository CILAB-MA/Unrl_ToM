
import numpy as np
import gym, yaml, os, sys, argparse
from gym.envs.registration import register
sys.path.append(os.getcwd())
from dipbluebot.dipblue import DipBlue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-simulation', default=1, type=int)
    parser.add_argument('--map-type', default='standard', type=str)
    parser.add_argument('--save-path', default='dipblue_analyze', type=str)
    args = parser.parse_args()
    return args

class DataAnalyzeTool(object):

    def __init__(self, map_type, save_path):

        self.map_type = map_type
        self.config_path = "./configs/{}.yaml".format(self.map_type)
        register(
            id='PressDiplomacyEnv-v0',
            entry_point='environment.env:PressDiplomacyEnv',
            kwargs={'config_path': self.config_path}
        )
        self.env = gym.make('PressDiplomacyEnv-v0')
        with open(self.config_path) as f:
            self.map_configs = yaml.load(f, Loader=yaml.FullLoader)
        self.save_path = save_path

    def run_env(self, num_simulation, sampled_weights):
        stats = dict()
        for nth_simulation in range(num_simulation):
            stat = self.play_one_epi(sampled_weights, self.map_configs['num_step'])
            self._data_extract(nth_simulation, stat)

    def play_one_epi(self, sampled_weights, num_step):
        self.env.reset()
        agents = {power: DipBlue(self.env.static_infos, power, weights=weight) for power, weight in
                  zip(list(self.env.static_infos['powers']), sampled_weights)}
        n = 0
        stat = {power:[] for power in self.env.static_infos['powers']}

        obs, infos = self.env.reset()
        while self.env.is_done() is False:

            start_phase = self.env.game.phase.split(" ")[0] + self.env.game.phase.split(" ")[-1]

            while self.env.game.phase.split(" ")[0] + self.env.game.phase.split(" ")[-1] == start_phase:

                if not self.env.is_nego:
                    for power, agent in agents.items():
                        power_orders, _, prev_order_clear = agent.act(infos)
                        self.env.submit((power, power_orders), prev_order_clear)

                else:
                    for power, agent in agents.items():
                        negos, agreed_orders, _ = agent.act(infos)
                        self.env.submit(negos, agreed_orders)

                obs, rew, done, infos = self.env.step(None)

            if n == num_step:
                break
            print(infos.keys())
            if (infos['name'][-1] == 'M'):
                for power in list(self.env.static_infos['powers']):
                    stat[power].append(dict(step=n, received=[], agreed=[], rejected=[], executed_order=dict()))
                n += 1

        return stat

    def run_data(self, data):
        pass

    def _data_extract(self,):
        pass

    def visualize(self):
        pass

if __name__ == '__main__':
    args = parse_args()
    analyzer = DataAnalyzeTool(args.map_type, args.save_path)

    analyzer.run_env(args.num_simulation, [[0.1, 0.9]] * 7)
