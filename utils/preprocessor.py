import argparse
import os
import numpy as np
import pickle
import random
from log_get_state import Np_state
import gym
from gym.envs.registration import register

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', '-et', type=int, default=1)  # 1:  기본 env, 2: small env
    parser.add_argument('--num_train_epi', '-nt', type=int, default=10)
    parser.add_argument('--num_eval_epi', '-ne', type=int, default=10)
    parser.add_argument('--num_past', '-np', type=int, default=3)
    parser.add_argument('--num_curr_step', '-nc', type=int, default=50)

    parser.add_argument('--num_log', '-nl', type=int, default=10)  # log 뽑을 때
    parser.add_argument('--num_sub_log', '-ns', type=int, default=5)  # log 뽑을 때
    parser.add_argument('--log_path', '-lp', type=str, default='./log_data/agent_30_id_99/')



    # parser.add_argument('--num_same_population_episode', '-np', type=int, default=5)  # 총 nt * np 만큼의 에피소드가 생성됨
    # parser.add_argument('--base_dir', '-b', type=str, default='./log_data')
    # parser.add_argument('--proc_dir', '-pd', type=str, default='./preprocessed_data')
    # parser.add_argument('--number', '-n', type=int, default=99)
    # parser.add_argument('--num_step', '-ns', type=int, default=200)
    # parser.add_argument('--num_cpu', '-p', type=int, default=40)
    args = parser.parse_args()
    return args


class Storage(object):

    def __init__(self, args, num_past, curr_step=40, base_dir=None, num_same_population_episode=5):
        self.args = args
        self.num_cur_step = curr_step
        msg_shape = (4, 76)
        num_step = args.num_step

        if args.env_type == 1:  # 7 players
            self.num_agent = 7
            self.obss_shape = (81, 75)
            self.num_weights = 3
        elif args.env_type == 2:  # samll env
            self.num_agent = 3
            self.obss_shape = (28, 47)
            self.num_weights = 3

        self.past_state = np.zeros([num_past, num_step, self.obss_shape[0], self.obss_shape[1]], dtype=np.int16)
        self.past_message = np.zeros([num_past, num_step, *msg_shape], dtype=np.int16)
        self.past_padding = np.zeros([num_past, num_step, 4], dtype=np.int16)  # todo
        self.curr_state = np.zeros([curr_step, self.obss_shape[0], self.obss_shape[1]], dtype=np.int16)
        self.curr_message = np.zeros([curr_step, *msg_shape], dtype=np.int16)
        self.curr_padding = np.zeros([curr_step, 1], dtype=np.int16)
        self.target_response = np.zeros([1], dtype=np.int16)
        self.agent_weights = np.zeros([num_past, self.num_agent, self.num_weights], dtype=np.int16)
        self.sender_index = np.zeros([num_past, 1], dtype=np.int16)
        self.receiver_index = np.zeros([num_past, 1], dtype=np.int16)
        # self.weights = weights
        # self.num_population = len(weights)
        self.num_past = num_past
        self.num_step = num_step
        self.num_same_population_episode = num_same_population_episode

    def extract(self, episodes, output_dir):

        past_current_idx = np.random.choice(np.arange(self.num_same_population_episode), size=self.num_past + 1,
                                            replace=False)

        send_ind = 0
        recv_ind = 1
        for past_epi in range(self.num_past + 1):
            # past episode
            with open(
                    f"{args.proc_dir}/get_state/agent_{self.num_agent}_id_{self.args.number}/{episodes}-{past_current_idx[past_epi]}.pickle",
                    'rb') as f:
                (sampled_weights, shuffle), obss = pickle.load(f)
                for send_idx in range(self.num_powers):
                    for recv_idx in range(self.num_powers):
                        if send_idx == recv_idx:
                            continue
                        with open(
                                f"{args.proc_dir}/get_state/agent_{self.num_agent}_id_{self.args.number}/{episodes}-{past_current_idx[past_epi]}[{send_ind}_to_{recv_ind}].pickle",
                                'rb') as f:
                            msg_list = pickle.load(f)

                        if past_epi != self.num_past:
                            self.agent_weights[past_epi] = sampled_weights[shuffle]
                            self.past_state[past_epi] = obss
                            for step in range(self.num_step):
                                self.past_message[past_epi, step] = msg_list[step][0]
                                self.past_padding[past_epi, step] = msg_list[step][2]



                        # current episode
                        else:
                            tmp_state, tmp_msg, tmp_send, tmp_target, tmp_step = [], [], [], [], []

                            for step in range(self.num_step):
                                all_msgs, send_padding, masking, sender_response = msg_list[step]
                                num_msg = len(np.where(send_padding == True)[0])
                                tmp_state.append(obss)
                                tmp_msg.append(all_msgs)

                                if num_msg != 0:
                                    tmp_step.append(step)
                                    tmp_send.append(send_ind)
                                    tmp_target.append(sender_response[send_ind])

                            if len(tmp_step) == 0:
                                self.curr_state = tmp_state[-self.num_cur_step:]
                                self.curr_message = tmp_msg[-self.num_cur_step:]
                                self.target_response = 2  # No Message
                            else:
                                ind = np.random.randint(0, len(tmp_step))
                                curr_step = tmp_step[ind] + 1
                                if curr_step < self.num_cur_step:
                                    self.curr_state[-curr_step:] = tmp_state[: curr_step]
                                    self.curr_message[-curr_step:] = tmp_msg[: curr_step]
                                else:
                                    self.curr_state = tmp_state[curr_step - self.num_cur_step: curr_step]
                                    self.curr_message = tmp_msg[curr_step - self.num_cur_step: curr_step]
                                self.curr_message[-1, self.num_cur_step - 1:] = 0
                                self.target_response = tmp_target[ind]

                            np.save(
                                f'{output_dir}/agent_weights_{episodes}-{past_current_idx[past_epi]}[{send_ind}_to_{recv_ind}].npy',
                                sampled_weights)
                            np.save(
                                f'{output_dir}/curr_message_{episodes}-{past_current_idx[past_epi]}[{send_ind}_to_{recv_ind}].npy',
                                self.curr_message)
                            np.save(
                                f'{output_dir}/curr_state_{episodes}-{past_current_idx[past_epi]}[{send_ind}_to_{recv_ind}].npy',
                                self.curr_state)
                            np.save(
                                f'{output_dir}/past_message_{episodes}-{past_current_idx[past_epi]}[{send_ind}_to_{recv_ind}].npy',
                                self.past_message)
                            np.save(
                                f'{output_dir}/past_state_{episodes}-{past_current_idx[past_epi]}[{send_ind}_to_{recv_ind}].npy',
                                self.past_state)
                            np.save(
                                f'{output_dir}/receiver_index_{episodes}-{past_current_idx[past_epi]}[{send_ind}_to_{recv_ind}].npy',
                                recv_ind)
                            np.save(
                                f'{output_dir}/sender_index_{episodes}-{past_current_idx[past_epi]}[{send_ind}_to_{recv_ind}].npy',
                                send_ind)
                            np.save(
                                f'{output_dir}/target_response_{episodes}-{past_current_idx[past_epi]}[{send_ind}_to_{recv_ind}].npy',
                                self.target_response)



if __name__ == "__main__":
    args = parse_args()

    register(
        id='PressDiplomacyEnv-v0',
        entry_point='environment.env:PressDiplomacyEnv')
    env = gym.make('PressDiplomacyEnv-v0')
    env.game_type = "standard" if args.env_type == 1 else "small"

    if args.env_type == 1:  # 7 players
        num_powers = 7
        obss_shape = (81, 75)
        msg_shape = (4, 76)
    elif args.env_type == 2:  # samll env
        num_powers = 3
        obss_shape = (28, 47)
        msg_shape = (4, 36)
    num_phase = 200
    np_state = Np_state(num_powers, env.static_infos)

    # train sample 뽑기
    for epi in range(args.num_train_epi):
        epi_idx_1 = random.randint(0, args.num_log - 1)
        # sample 한 세트 뽑기
        for past_epi in range(args.num_past + 1):
            epi_idx_2 = random.randint(0, args.num_sub_log - 1)
            picked_epi_path = args.log_path + "epi_{}-{}.pickle".format(epi_idx_1, epi_idx_2)
            with open(picked_epi_path, 'rb') as f:
                (sampled_weight, shuffled_idx), log = pickle.load(f)

            sender_idx, receiver_idx = random.sample(range(num_powers), 2)

            obss = np.zeros([num_phase, *obss_shape])
            message = [[np.zeros([*msg_shape]),
                        np.full((4,), False),
                        np.full((4,), False),
                        np.full((4,), 2)] for _ in range(num_phase)]

            for i in range(num_phase):
                if log['infos'][i]['name'] == 'COMPLETED':
                    break
                obss[i] = np.concatenate([np_state.prev_board_state_maker(log, i),
                                          np_state.prev_order_state_maker(log, i)], axis=-1)










