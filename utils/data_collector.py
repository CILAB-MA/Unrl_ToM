import argparse
import sys
sys.path.append('./')
from environment.wrapper import PressDiplomacyWrapper
from utils import *
from storage import Storage
import numpy as np
import os, copy, math
import gym
from gym.envs.registration import register
import multiprocessing as mp
import time

register(
    id='PressDiplomacyEnv-v0',
    entry_point='environment.env:PressDiplomacyEnv')

def get_bool(args):
    return eval(args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_idx', '-ei', type=int, default=2)  # 1 기본, 2 small
    parser.add_argument('--num_agent', '-na', type=int, default=7)
    parser.add_argument('--num_episode', '-e', type=int, default=1)
    parser.add_argument('--num_eval', '-ev', type=int, default=0)
    parser.add_argument('--number', '-n', type=int, default=99)
    parser.add_argument('--num_past', '-np', type=int, default=1)
    parser.add_argument('--num_step', '-ns', type=int, default=100)
    parser.add_argument('--alpha', '-a', type=float, default=0.01)
    parser.add_argument('--base_dir', '-b', type=str, default='/data')
    parser.add_argument('--curr_step', '-c', type=int, default=1)
    parser.add_argument('--data_type', '-w', type=str, default='full')
    # parser.add_argument('--num_episode', '-e', type=int, default=30)
    parser.add_argument('--fixed_weight', '-fw', type=float, default=[math.pi/4])  # 꼭 넣을 weight
    parser.add_argument('--num_cpu', '-p', type=int, default=1)
    args = parser.parse_args()
    return args


class DataCollector(object):

    def __init__(self, args):

        # make settings

        if args.num_agent == 1:
            self.population_weights = args.fixed_weight * args.num_agent
            self.diff_population_weights = args.fixed_weight * args.num_agent
        else:
            self.population_weights = make_weights(args.num_agent)
            self.population_weights = np.concatenate([self.population_weights, args.fixed_weight])
            self.diff_population_weights = make_weights(args.num_agent)
            self.diff_population_weights = np.concatenate([self.diff_population_weights, args.fixed_weight])

        foldername = 'agent_{}_id_{}_{}'.format(args.num_agent, args.number, args.ncpu)

        self.env = PressDiplomacyWrapper(gym.make('PressDiplomacyEnv-v0'))
        self.env.game_type = "small"
        self.storage = Storage(self.env, args.env_idx, self.population_weights, args.num_past, args.num_episode, args.num_step, args.curr_step)
        self.diff_storage = Storage(self.env, args.env_idx, self.diff_population_weights, args.num_past, args.num_eval, args.num_step, args.curr_step)

        self.base_dir = os.path.join(args.base_dir, foldername)

    def make_train_set(self):
        self.tr_data = self.storage.extract()
        save_data(self.tr_data, 'train', self.base_dir)

    def make_eval_set(self):
        eval_type1_data = self.diff_storage.extract()
        save_data(eval_type1_data, 'eval', self.base_dir, eval=1)

def concat(args):
    base_dir = args.base_dir
    exp = f"agent_{args.num_agent}_id_{args.number}"
    folder_list =  [base_dir + '/' + f for f in os.listdir(base_dir) if f.startswith(exp)]

    FILE_LIST = ['agent_weights.npy', 'curr_message.npy', 'curr_state.npy', 'past_message.npy', 
                'past_state.npy', 'receiver_index.npy', 'sender_index.npy', 'target_response.npy']
    
    agent_weight_train_total, agent_weight_eval_total = [], []
    curr_message_train_total, curr_message_eval_total = [], []
    curr_state_train_total, curr_state_eval_total = [], []
    past_message_train_total, past_message_eval_total = [], []
    past_state_train_total, past_state_eval_total = [], []
    receiver_index_train_total, receiver_index_eval_total = [], []
    sender_index_train_total, sender_index_eval_total = [], []
    target_response_train_total, target_response_eval_total = [], []

    for f in folder_list:
        agent_weight_train_total.append(np.load(f + '/train' + '/agent_weights.npy'))
        agent_weight_eval_total.append(np.load(f + '/eval/1' + '/agent_weights.npy'))
        curr_message_train_total.append(np.load(f + '/train' + '/curr_message.npy'))
        curr_message_eval_total.append(np.load(f + '/eval/1' + '/curr_message.npy'))
        curr_state_train_total.append(np.load(f + '/train' + '/curr_state.npy'))
        curr_state_eval_total.append(np.load(f + '/eval/1' + '/curr_state.npy'))
        past_message_train_total.append(np.load(f + '/train' + '/past_message.npy'))
        past_message_eval_total.append(np.load(f + '/eval/1' + '/past_message.npy'))
        past_state_train_total.append(np.load(f + '/train' + '/past_state.npy'))
        past_state_eval_total.append(np.load(f + '/eval/1' + '/past_state.npy'))
        receiver_index_train_total.append(np.load(f + '/train' + '/receiver_index.npy'))
        receiver_index_eval_total.append(np.load(f + '/eval/1' + '/receiver_index.npy'))
        sender_index_train_total.append(np.load(f + '/train' + '/sender_index.npy'))
        sender_index_eval_total.append(np.load(f + '/eval/1' + '/sender_index.npy'))
        target_response_train_total.append(np.load(f + '/train' + '/target_response.npy'))
        target_response_eval_total.append(np.load(f + '/eval/1' + '/target_response.npy'))
    
    try:
        os.makedirs(base_dir + '/' + 'total_'+ exp + '/train')
        os.makedirs(base_dir + '/' + 'total_'+ exp + '/eval/1')
    except FileExistsError:
        print("folder for the total already exists")
    
    train_dir = base_dir + '/' + 'total_'+ exp + '/train'
    eval_dir = base_dir + '/' + 'total_'+ exp + '/eval/1'

    np.save(train_dir + '/agent_weights.npy', np.array(agent_weight_train_total))
    np.save(eval_dir + '/agent_weights.npy', np.array(agent_weight_eval_total))
    np.save(train_dir + '/curr_message.npy', np.array(curr_message_train_total))
    np.save(eval_dir + '/curr_message.npy', np.array(curr_message_eval_total))
    np.save(train_dir + '/curr_state.npy', np.array(curr_state_train_total))
    np.save(eval_dir + '/curr_state.npy', np.array(curr_state_eval_total))
    np.save(train_dir + '/past_message.npy', np.array(past_message_train_total))
    np.save(eval_dir + '/past_message.npy', np.array(past_message_eval_total))
    np.save(train_dir + '/past_state.npy', np.array(past_state_train_total))
    np.save(eval_dir + '/past_state.npy', np.array(past_state_eval_total))
    np.save(train_dir + '/receiver_index.npy', np.array(receiver_index_train_total))
    np.save(eval_dir + '/receiver_index.npy', np.array(receiver_index_eval_total), )
    np.save(train_dir + '/sender_index.npy', np.array(sender_index_train_total))
    np.save(eval_dir + '/sender_index.npy', np.array(sender_index_eval_total))
    np.save(train_dir + '/target_response.npy', np.array(target_response_train_total))
    np.save(eval_dir + '/target_response.npy', np.array(target_response_eval_total))


def main(n):
    import time;import random
    args = parse_args()
    args.ncpu = n
    collector = DataCollector(args)
    seed = (os.getpid() * int(time.time())) % 1234
    np.random.seed(seed)
    random.seed(seed)
    collector.make_train_set()
    # collector.make_eval_set()

if __name__ == '__main__':
    start = time.perf_counter()
    args = parse_args()
    num_cpu = args.num_cpu
    with mp.Pool(processes = num_cpu) as p:
        p.map(main, range(num_cpu))
    # concat(args)
    print(f"Elapsed time : {time.perf_counter() - start}")