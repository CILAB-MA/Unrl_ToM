import numpy as np
import copy
from tqdm import tqdm
import math
import gym
from dipbluebot.dipblue import DipBlue
from environment.get_state import *

class Storage(object):

    def __init__(self, env, env_idx, weights, num_past, num_episode,
                 num_step, curr_step=40):
        self.env = env
        self.num_cur_step = curr_step
        msg_shape = (4, 76)

        if env_idx == 1:  # 7 players
            self.num_powers = 7
            self.obss_shape = (81, 75)
            self.msg_shape = (4, 76)
        elif env_idx == 2:  # samll env
            self.num_powers = 3
            self.obss_shape = (28, 47)
            self.msg_shape = (4, 36)

        self.past_state = np.zeros([num_episode, num_past, num_step, self.obss_shape[0], self.obss_shape[1]], dtype=np.int16)
        self.past_message = np.zeros([num_episode, num_past, num_step, *self.msg_shape], dtype=np.int16)
        self.past_padding = np.zeros([num_episode, num_past, num_step, 4], dtype=np.int16) # todo
        self.curr_state = np.zeros([num_episode, curr_step, self.obss_shape[0], self.obss_shape[1]], dtype=np.int16)
        self.curr_message = np.zeros([num_episode, curr_step, *self.msg_shape], dtype=np.int16)
        self.curr_padding = np.zeros([num_episode, curr_step, 1], dtype=np.int16)
        self.target_response = np.zeros([num_episode, 1], dtype=np.int16)
        self.agent_weights = np.zeros([num_episode, num_past, self.num_powers], dtype=np.int16)
        self.sender_index = np.zeros([num_episode, num_past, 1], dtype=np.int16)
        self.receiver_index = np.zeros([num_episode, num_past, 1], dtype=np.int16)
        self.weights = weights
        self.num_population = len(weights)
        self.num_past = num_past
        self.num_step = num_step
        self.num_episode = num_episode

    def extract(self):
        for epi in tqdm(range(self.num_episode), total=self.num_episode):
            sampled_weights = np.random.randint(0, self.num_population, size=self.num_powers - 1)
            other = np.random.choice(sampled_weights, 1)
            sampled_weights = np.concatenate([sampled_weights, [self.num_population - 1]])
            for past_epi in range(self.num_past + 1):
                np.random.shuffle(sampled_weights)
                recv_candidates = np.where(sampled_weights==other)[0]
                recv_power = np.random.choice(recv_candidates, 1)[0]
                send_candidates = np.where(sampled_weights==(len(self.weights) - 1))[0]
                send_power = np.random.choice(send_candidates, 1)[0]

                obss, msgs = self.env.reset(self.weights[sampled_weights], send_power, recv_power)
                all_msgs, send_padding, masking, sender_response = msgs

                # collect the past traj
                if past_epi != self.num_past:
                    self.sender_index[epi, past_epi] = send_power
                    self.receiver_index[epi, past_epi] = recv_power
                    self.agent_weights[epi, past_epi] = sampled_weights
                    for step in range(self.num_step):
                        self.past_state[epi, past_epi, step] = obss
                        self.past_message[epi, past_epi, step] = all_msgs

                        obss, msgs, done = self.env.step()
                        all_msgs, send_padding, masking, sender_response = msgs

                        self.past_state[epi, past_epi, step] = obss
                        self.past_message[epi, past_epi, step] = all_msgs
                        self.past_padding[epi, past_epi, step] = masking

                        if done:
                            break

                # collect the current traj
                else:
                    tmp_state = []
                    tmp_msg = []
                    tmp_send = []
                    tmp_target = []
                    tmp_step = []
                    for step in range(self.num_step):
                        num_msg = len(np.where(send_padding == True)[0])
                        tmp_state.append(obss)
                        tmp_msg.append(all_msgs)

                        if num_msg != 0:
                            send_ind = np.random.randint(num_msg)
                            tmp_step.append(step)
                            tmp_send.append(send_ind)
                            tmp_target.append(sender_response[send_ind])

                        obss, msgs, done = self.env.step()
                        all_msgs, send_padding, masking, sender_response = msgs


                    if len(tmp_step) == 0:
                        self.curr_state[epi] = tmp_state[-self. ur_step:]
                        self.curr_message[epi] = tmp_msg[-self.num_cur_step:]
                        self.target_response[epi] = 2 # No Message
                    else:
                        ind = np.random.randint(0, len(tmp_step))
                        curr_step = tmp_step[ind] + 1
                        if curr_step < self.num_cur_step:
                            self.curr_state[epi, -curr_step:] = tmp_state[: curr_step]
                            self.curr_message[epi, -curr_step:] = tmp_msg[: curr_step]
                        else:
                            self.curr_state[epi] = tmp_state[curr_step - self.num_cur_step: curr_step]
                            self.curr_message[epi] = tmp_msg[curr_step - self.num_cur_step: curr_step]
                        self.curr_message[epi, -1, self.num_cur_step - 1:] = 0
                        self.target_response[epi] = tmp_target[ind]
        return dict(past_state=self.past_state,
                    past_message=self.past_message,
                    curr_state=self.curr_state,
                    curr_message=self.curr_message,
                    target_response=self.target_response,
                    agent_weights=self.agent_weights,
                    sender_index=self.sender_index,
                    receiver_index=self.receiver_index
                    )

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