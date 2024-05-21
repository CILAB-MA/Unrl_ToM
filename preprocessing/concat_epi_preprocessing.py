import argparse
import os, copy, sys
sys.path.append(os.getcwd())
import random
import numpy as np
import multiprocessing as mp
import pickle as pkl
from regex import F
from tqdm import tqdm
from utils import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', '-et', type=int, default=1)  # 1:  기본 env, 2: small env
    parser.add_argument('--num_same_population_episode', '-nsp', type=int, default=5)  # 총 nt * np 만큼의 에피소드가 생성됨
    parser.add_argument('--num_population', '-np', type=int, default=30)
    parser.add_argument('--proc_dir', '-pd', type=str, default='')
    parser.add_argument('--number', '-n', type=int, default=99)
    parser.add_argument('--num_cpu', '-p', type=int, default=40)
    args = parser.parse_args()
    return args

class Storage(object):

    def __init__(self, args, num_step, curr_step=40, base_dir = None):
        self.args = args
        self.num_cur_step = curr_step
        self.num_step = num_step
        self.num_past = args.num_same_population_episode-1
        if args.env_type == 1:  # 7 players
            self.num_powers = 7
            self.internal_shape = (3, )
            self.order_shape = (181,)
            self.target_order_shape = (4, )
            self.input_src_shape = (81, )
            self.target_dst_shape = (81, )
            self.map_tactician_shape = (120, )
            self.msg_shape = (2 * (5 * self.num_powers + 8 + 2))
            self.board_shape = (num_loc, num_agent * 3 + 14)
            self.msg_mask_shape = (40, )
            self.num_weights = 4

        elif args.env_type == 2:  # samll env
            # TODO : change the several numbers to run the small env
            self.num_powers = 3
            self.internal_shape = (12, )
            self.order_shape = (71,)
            self.target_order_shape = (4, )
            self.input_src_shape = (28, )
            self.target_dst_shape = (28, )
            self.map_tactician_shape = (42, )
            self.num_weights = 4

        self.past_me_internal_state = np.zeros([self.num_past, num_step, self.internal_shape[0]], dtype=np.float16)
        self.past_me_map_tactician = np.zeros([self.num_past, num_step, self.map_tactician_shape[0]], dtype=np.int32)
        self.past_other_internal_state = np.zeros([self.num_past, num_step, self.internal_shape[0]], dtype=np.float16)
        self.past_other_map_tactician = np.zeros([self.num_past, num_step, self.map_tactician_shape[0]], dtype=np.int32)
        self.past_order = np.zeros([self.num_past, num_step, self.order_shape[0]], dtype=np.uint8)
        self.past_msg_state = np.zeros([self.num_past, num_step, self.msg_shape], dtype=np.uint8)
        self.past_board_state = np.zeros([self.num_past, num_step, self.board_shape[0], self.board_shape[1]], dtype=np.uint8)

        self.curr_me_internal_state = np.zeros([curr_step, self.internal_shape[0]], dtype=np.float16)
        self.curr_me_map_tactician = np.zeros([curr_step, self.map_tactician_shape[0]], dtype=np.int32)
        self.curr_other_internal_state = np.zeros([curr_step, self.internal_shape[0]], dtype=np.int32)
        self.curr_other_map_tactician = np.zeros([curr_step, self.map_tactician_shape[0]], dtype=np.int32)
        self.curr_msg_state = np.zeros([curr_step, self.msg_shape], dtype=np.uint8)
        self.curr_order = np.zeros([curr_step - 1, self.order_shape[0]], dtype=np.uint8)
        self.curr_board_state = np.zeros([curr_step, self.board_shape[0], self.board_shape[1]], dtype=np.uint8)

        self.curr_src = np.zeros([self.input_src_shape[0]], dtype=np.uint8)


        self.target_order = np.zeros([1], dtype=np.uint8)
        self.target_dst = np.zeros([1], dtype=np.uint8)
        self.target_recv = np.zeros([1], dtype=np.uint8)
        self.me_weights = np.zeros([self.num_weights], dtype=np.float16)
        self.other_weights = np.zeros([self.num_weights], dtype=np.float16)
        self.sender_index = np.zeros([self.num_past, 1], dtype=np.uint8)
        self.receiver_index = np.zeros([self.num_past, 1], dtype=np.uint8)
        # self.weights = weights
        # self.num_population = len(weights)
    def reduce_message(self, messages, send_me):
        # send_me : if 1, me send, else other send (200, 40)
        # output : concat(sum of me send, sum of other send)
        send_me = np.expand_dims(send_me,axis=-1)
        send_me = send_me.repeat(45, axis=-1)
        recv_me = abs(send_me - 1)
        send_messages = send_me * messages
        recv_messages = recv_me * messages
        send_messages = send_messages.sum(1, keepdims=True).squeeze(1)
        recv_messages = recv_messages.sum(1, keepdims=True).squeeze(1)
        added = np.concatenate([send_messages, recv_messages], axis=-1)
        return added.astype(np.uint8)

    def extract(self, episodes, output_dir):
        #nt 당 데이터 하나
        past_current_idx = np.random.permutation(np.arange(self.args.num_same_population_episode))

        candidates = []
        for x in range(self.num_powers):
            for y in range(self.num_powers):
                if x == y :
                    continue
                candidates.append((x,y))

        mes = np.zeros((self.num_past + 1), dtype=np.uint8)
        others = np.zeros((self.num_past + 1), dtype=np.uint8)
        one_msg_shape = self.msg_shape - 2 # 88
        for past_epi in range(self.num_past + 1):
            # past episode
            with open(f"{args.proc_dir}/get_state/agent_{num_population}_id_{self.args.number}/"
                      f"{episodes}_{past_current_idx[past_epi]}.pickle", 'rb') as f:
                (me, other, epi_step), other_map_tactician, other_internal_states, \
                        me_map_tactician, me_internal_states, board_states, orders_target, orders_src, \
                        orders_dst, src_powers, dst_powers, orders_len = pkl.load(f)
            me_idx, me_power, me_weight = me
            other_idx, other_power, other_weight = other
            # make order
            order = np.concatenate([orders_target, orders_src, orders_dst, src_powers, dst_powers], axis=-1)
            order = np.sum(order, axis=1, dtype=np.uint8)

            with open(f"{args.proc_dir}/get_state/agent_{num_population}_id_{self.args.number}/"
                        f"{episodes}_{past_current_idx[past_epi]}[{me[0]}_to_{other[0]}].pickle", 'rb') as f:
                msg_list, msg_len, me_send = pkl.load(f)

            if past_epi != self.num_past:
                self.past_order[past_epi] = order[:self.num_step]
                self.past_me_map_tactician[past_epi] = me_map_tactician[:self.num_step]
                self.past_me_internal_state[past_epi] = me_internal_states[:self.num_step]
                self.past_other_map_tactician[past_epi] = other_map_tactician[:self.num_step]
                self.past_other_internal_state[past_epi] = other_internal_states[:self.num_step]
                self.past_msg_state[past_epi] = self.reduce_message(msg_list[:self.num_step], me_send)
                self.past_board_state[past_epi] = board_states[:self.num_step]

            # current episode
            else:
                order_nonzero_step = np.where(orders_len != 0)[0]
                msg_nonzero_step = np.where(msg_len != 0)[0]
                both_nonzero_step = np.intersect1d(msg_nonzero_step, order_nonzero_step)
                both_nonzero_step = both_nonzero_step[both_nonzero_step != 0]
                chosen_step = np.random.choice(both_nonzero_step, 1)[0]
                
                # choose src unit
                ith_order = np.random.randint(0, orders_len[chosen_step], 1)
                jth_candidates = np.where(me_send[chosen_step] == 1)[0]
                jth_message = np.random.choice(jth_candidates, 1)[0]

                self.input_src = orders_src[chosen_step, ith_order]
                #self.input_send = msg_list[chosen_step, jth_message][:one_msg_shape]
                self.input_send = msg_list[chosen_step, jth_message][:43]
                self.target_dst = orders_dst[chosen_step, ith_order]
                #recv = msg_list[chosen_step, jth_message][one_msg_shape:]
                recv = msg_list[chosen_step, jth_message][43:]
                self.target_order = orders_target[chosen_step, ith_order]

                if chosen_step < self.num_cur_step:
                    self.curr_me_internal_state[-chosen_step - 1:] = me_internal_states[:chosen_step + 1]
                    self.curr_me_map_tactician[-chosen_step - 1:] = me_map_tactician[:chosen_step + 1]
                    self.curr_other_internal_state[-chosen_step - 1:] = other_internal_states[:chosen_step + 1]
                    self.curr_other_map_tactician[-chosen_step - 1:] = other_map_tactician[:chosen_step + 1]
                    self.curr_msg_state[-chosen_step - 1:] = self.reduce_message(msg_list[:chosen_step + 1],
                                                                                    me_send[:chosen_step + 1])
                    self.curr_order[-chosen_step:] = order[:chosen_step] # action has 39 steps, plz check for it!
                    self.curr_board_state[-chosen_step - 1:] = board_states[:chosen_step + 1]
                else:
                    self.curr_me_internal_state = me_internal_states[chosen_step - self.num_cur_step + 1: chosen_step + 1]
                    self.curr_me_map_tactician = me_map_tactician[chosen_step - self.num_cur_step + 1: chosen_step + 1]
                    self.curr_other_internal_state = other_internal_states[chosen_step - self.num_cur_step + 1: chosen_step + 1]
                    self.curr_other_map_tactician = other_map_tactician[chosen_step - self.num_cur_step + 1: chosen_step + 1]
                    self.curr_msg_state = self.reduce_message(msg_list[chosen_step - self.num_cur_step + 1: chosen_step + 1],
                                                                me_send[chosen_step - self.num_cur_step + 1: chosen_step + 1])
                    self.curr_order = order[chosen_step - self.num_cur_step + 1: chosen_step]
                    self.curr_board_state = board_states[chosen_step - self.num_cur_step + 1: chosen_step + 1]

                self.target_recv = np.where(recv == 1)[0]
                mes[past_epi] = me_idx
                others[past_epi] = other_idx
                self.me_weights = me_weight
                self.other_weights = other_weight
            
        np.save(f'{output_dir}/me_weights_{episodes}-{past_current_idx[past_epi]}.npy', self.me_weights)
        np.save(f'{output_dir}/other_weights_{episodes}-{past_current_idx[past_epi]}.npy', self.other_weights)
        np.save(f'{output_dir}/curr_me_map_tactician_{episodes}-{past_current_idx[past_epi]}.npy', self.curr_me_map_tactician)
        np.save(f'{output_dir}/curr_other_map_tactician_{episodes}-{past_current_idx[past_epi]}.npy', self.curr_other_map_tactician)
        np.save(f'{output_dir}/curr_src_{episodes}-{past_current_idx[past_epi]}.npy', self.input_src)
        np.save(f'{output_dir}/curr_send_{episodes}-{past_current_idx[past_epi]}.npy', self.input_send)
        np.save(f'{output_dir}/curr_me_internal_{episodes}-{past_current_idx[past_epi]}.npy', self.curr_me_internal_state)
        np.save(f'{output_dir}/curr_other_internal_{episodes}-{past_current_idx[past_epi]}.npy', self.curr_other_internal_state)
        np.save(f'{output_dir}/curr_order_{episodes}-{past_current_idx[past_epi]}.npy', self.curr_order)
        np.save(f'{output_dir}/curr_message_{episodes}-{past_current_idx[past_epi]}.npy', self.curr_msg_state)
        np.save(f'{output_dir}/curr_board_{episodes}-{past_current_idx[past_epi]}.npy', self.curr_board_state)
        np.save(f'{output_dir}/past_me_internal_{episodes}-{past_current_idx[past_epi]}.npy', self.past_me_internal_state)
        np.save(f'{output_dir}/past_other_internal_{episodes}-{past_current_idx[past_epi]}.npy', self.past_other_internal_state)
        np.save(f'{output_dir}/past_me_map_tactician_{episodes}-{past_current_idx[past_epi]}.npy', self.past_me_map_tactician)
        np.save(f'{output_dir}/past_other_map_tactician_{episodes}-{past_current_idx[past_epi]}.npy', self.past_other_map_tactician)
        np.save(f'{output_dir}/past_order_{episodes}-{past_current_idx[past_epi]}.npy', self.past_order)
        np.save(f'{output_dir}/past_message_{episodes}-{past_current_idx[past_epi]}.npy', self.past_msg_state)
        np.save(f'{output_dir}/past_board_{episodes}-{past_current_idx[past_epi]}.npy', self.past_board_state)
        np.save(f'{output_dir}/other_index_{episodes}-{past_current_idx[past_epi]}.npy', other)
        np.save(f'{output_dir}/me_index_{episodes}-{past_current_idx[past_epi]}.npy', mes)
        np.save(f'{output_dir}/other_index_{episodes}-{past_current_idx[past_epi]}.npy', others)
        np.save(f'{output_dir}/target_order_{episodes}-{past_current_idx[past_epi]}.npy', self.target_order)
        np.save(f'{output_dir}/target_dst_{episodes}-{past_current_idx[past_epi]}.npy', self.target_dst)
        np.save(f'{output_dir}/target_recv_{episodes}-{past_current_idx[past_epi]}.npy', self.target_recv)

    def concat(self, output_dir):
        files = ['me_weights', 'other_weights', 'curr_me_map_tactician', 'curr_other_map_tactician', 'curr_order',
                 'curr_src', 'curr_send', 'curr_me_internal', 'curr_other_internal', 'curr_board', 'curr_message',
                 'past_me_map_tactician', 'past_other_map_tactician', 'past_order', 'past_me_internal', 'past_board',
                 'past_message', 'past_other_internal', 'other_index', 'me_index',
                 'target_order', 'target_dst', 'target_recv']
        os.listdir(output_dir)
        for file in files:
            rows = [row for row in os.listdir(output_dir) if row.startswith(file)]
            nrow = len(rows)
            first = np.load(output_dir + '/' + rows[0])

            if ('internal' in file) or ('weights' in file):
                total = np.zeros([nrow, *first.shape], dtype=np.float16)
            elif ('map_tactician' in file):
                total = np.zeros([nrow, *first.shape], dtype=np.int32)
            else:
                total = np.zeros([nrow, *first.shape], dtype=np.uint8)
            for i, r in enumerate(rows):
                total[i] = np.load(output_dir + '/' + r)
                os.remove(output_dir + '/' + r)
            np.save(output_dir + '/' + file + '.npy', total)

def output(proc_id):
    workloads = [input_list[args.num_cpu * i + proc_id] for i in range(len(input_list)//args.num_cpu + 1) if args.num_cpu * i + proc_id < len(input_list)]

    for file in tqdm(workloads, desc= f'#{proc_id}', position=proc_id+1):
        episodes = file[:file.find(".")+1][:file.find("[")-2]
        preprocessor2.extract(episodes, output_dir)
        #print(proc_id)

if __name__=="__main__":
    args = parse_args()

    preprocessed_dir = args.proc_dir
    num_population = args.num_population
    num_agent = 7 if args.env_type == 1 else 3
    num_loc = 81 if args.env_type == 1 else 28
    num_step = 200 if args.env_type == 1 else 100

    order_shape = (num_loc, num_agent * 4 + 12)
    board_shape = (num_loc, num_agent * 3 + 14)
    run_id = args.number
    COUNTRIES = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"] if args.env_type==1 else ['FRANCE', 'GERMANY', 'ITALY']
    output_dir = preprocessed_dir + f"/storage/agent_{num_population}_id_{run_id}"
    utils.make_dirs(output_dir)

    preprocessor2 = Storage(args, num_step)

    input_list = os.listdir(preprocessed_dir + f"/get_state/agent_{num_population}_id_{run_id}")
    input_list = sorted(input_list)

    num_cpu = args.num_cpu
    try:
        with mp.Pool(processes = num_cpu) as p:
            p.map(output, range(num_cpu))
    except:
        from multiprocessing.pool import ThreadPool
        with ThreadPool(num_cpu) as p:
            p.map(output, range(num_cpu))

    preprocessor2.concat(output_dir)