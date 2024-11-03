import time
import random
import argparse, logging
from tqdm import tqdm
import sys, os
sys.path.append('')
sys.path.append(os.getcwd())
sys.path.append("/app/Dip_ToM/")
import gym
from gym.envs.registration import register
import multiprocessing as mp
from utils.utils import *
from log_storage import Storage
import os
import numpy as np
import logging
import traceback

def parse_args():
    parser = argparse.ArgumentParser()
    # standard, small, ancmed, pure, clonial, empire, known_world_901, modern, world
    parser.add_argument('--env_type', '-et', type=str, default="standard")
    parser.add_argument('--num_population', '-np', type=int, default=30)
    parser.add_argument('--population_pool_dir', '-pool', type=str, default='preprocessing/population_pool_30.pickle')
    parser.add_argument('--num_train_episode', '-nt', type=int, default=10)
    parser.add_argument('--num_same_population_episode', '-nsp', type=int, default=5)  # 총 nt * np 만큼의 에피소드가 생성됨
    parser.add_argument('--base_dir', '-b', type=str, default='./preprocessing/log_data')
    parser.add_argument('--number', '-n', type=int, default=0)
    parser.add_argument('--alpha', '-a', type=float, default=0.01)
    parser.add_argument('--data_type', '-w', type=str, default='full')
    parser.add_argument('--num_cpu', '-p', type=int, default=40)
    args = parser.parse_args()
    return args


class DataCollector(object):

    def __init__(self, args):
        self.population_pool = load_weights(args.population_pool_dir)
        folder_name = 'agent_{}_id_{}'.format(args.num_population, args.number)
        config_path = "./configs/{}.yaml".format(args.env_type)
        register(
            id='PressDiplomacyEnv-v0',
            entry_point='environment.env:PressDiplomacyEnv',
            kwargs={'config_path': config_path})
        self.env = gym.make('PressDiplomacyEnv-v0')
        self.storage = Storage(self.env, args.env_type, self.population_pool, args.num_train_episode,
                               args.num_same_population_episode)
        self.base_dir = os.path.join(args.base_dir, folder_name)

    def make_log_dataset(self, args):
        # Extract total (num_episode * num_sub_episode) episodes
        import time
        for epi in tqdm(range(args.num_train_episode)):
            # Select (num_powers) weights randomly
            sampled_weights = self.population_pool[np.random.randint(self.population_pool.shape[0], size=self.storage.num_powers), :]
            for sub_epi in range(args.num_same_population_episode):
                shuffle = np.random.permutation(np.arange(self.storage.num_powers))  #어떤 character가 메시지를 보냈는지 확인하기위해 index를 저장
                epi_log = self.storage.extract_epi(sampled_weights[shuffle])
                save_log_data(sampled_weights, shuffle, epi_log, self.base_dir, "/epi_{}_{}.npz".format(epi * args.num_cpu + args.ncpu, sub_epi))

def main(n):
    args = parse_args()
    args.ncpu = n
    collector = DataCollector(args)
    # seed = (os.getpid() * int(time.time())) % 1234
    # np.random.seed(seed)
    # random.seed(seed)
    collector.make_log_dataset(args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    start = time.perf_counter()
    args = parse_args()
    num_cpu = args.num_cpu
    try:
        with mp.Pool(processes = num_cpu) as p:
            p.map(main, range(num_cpu))
    except Exception as e:
        logging.basicConfig(filename=f'./test{args.number}_raw.log', level=logging.ERROR)
        logging.error(traceback.format_exc())
    print(f"Elapsed time : {time.perf_counter() - start}")

