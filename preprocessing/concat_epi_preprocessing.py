import argparse
import os, sys, yaml
sys.path.append(os.getcwd())
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from utils import utils
import time
import logging
import traceback

def parse_args():
    parser = argparse.ArgumentParser()
    # standard, small, ancmed, pure, clonial, empire, known_world_901, modern, world
    parser.add_argument('--env_type', '-et', type=str, default="standard")
    parser.add_argument('--num_same_population_episode', '-nsp', type=int, default=5)  # 총 nt * np 만큼의 에피소드가 생성됨
    parser.add_argument('--num_population', '-np', type=int, default=30)
    parser.add_argument('--proc_dir', '-pd', type=str, default='./preprocessing/preprocessed_data')
    parser.add_argument('--number', '-n', type=int, default=0)
    parser.add_argument('--num_cpu', '-p', type=int, default=40)
    parser.add_argument('--data_type', '-d', type=str, default="normal")  # normal, betray, fulfill 데이터 고르기
    args = parser.parse_args()
    return args

class Storage(object):

    def __init__(self, args, curr_step=40, base_dir = None):
        self.args = args
        with open('./configs/{}.yaml'.format(args.env_type)) as f:
            yd = yaml.load(f, Loader=yaml.FullLoader)
        self.num_step = yd['num_step']
        self.num_cur_step = int(self.num_step / 4)
        self.num_past = args.num_same_population_episode-1

        self.num_powers = yd['num_agent']
        self.internal_shape = (3, )
        self.order_shape = (5 + 2 * (yd['num_loc'] + yd['num_agent']),)
        self.target_order_shape = (4, )
        self.input_src_shape = (yd['num_loc'], )
        self.target_dst_shape = (yd['num_loc'], )
        self.map_tactician_shape = (yd['num_map_tact'], )
        self.msg_shape = (2 * (5 * self.num_powers + 8 + 2))
        self.max_action = yd['max_action']
        self.board_shape = (yd['num_loc'], yd['num_agent'] * 3 + 14)
        self.msg_mask_shape = (40, )
        self.num_weights = 2

        self.past_me_internal_state = np.zeros([self.num_past, self.num_step, self.internal_shape[0]], dtype=np.float16)
        self.past_me_map_tactician = np.zeros([self.num_past, self.num_step, self.map_tactician_shape[0]], dtype=np.int32)
        self.past_other_internal_state = np.zeros([self.num_past, self.num_step, self.internal_shape[0]], dtype=np.float16)
        self.past_other_map_tactician = np.zeros([self.num_past, self.num_step, self.map_tactician_shape[0]], dtype=np.int32)
        self.past_order = np.zeros([self.num_past, self.num_step, self.order_shape[0]], dtype=np.float16)
        self.past_msg_state = np.zeros([self.num_past, self.num_step, self.msg_shape], dtype=np.float16)
        self.past_board_state = np.zeros([self.num_past, self.num_step, self.board_shape[0], self.board_shape[1]], dtype=np.float16)

        self.curr_me_internal_state = np.zeros([self.num_cur_step, self.internal_shape[0]], dtype=np.float16)
        self.curr_me_map_tactician = np.zeros([self.num_cur_step, self.map_tactician_shape[0]], dtype=np.int32)
        self.curr_other_internal_state = np.zeros([self.num_cur_step, self.internal_shape[0]], dtype=np.int32)
        self.curr_other_map_tactician = np.zeros([self.num_cur_step, self.map_tactician_shape[0]], dtype=np.int32)
        self.curr_msg_state = np.zeros([self.num_cur_step, self.msg_shape], dtype=np.float16)
        self.curr_order = np.zeros([self.num_cur_step - 1, self.order_shape[0]], dtype=np.float16)
        self.curr_board_state = np.zeros([self.num_cur_step, self.board_shape[0], self.board_shape[1]], dtype=np.float16)

        self.curr_src = np.zeros([self.input_src_shape[0]], dtype=np.uint8)
        self.one_msg_shape = (5 * self.num_powers + 8)

        self.target_order = np.zeros([1], dtype=np.uint8)
        self.target_dst = np.zeros([1], dtype=np.uint8)
        self.target_recv = np.zeros([1], dtype=np.uint8)
        self.target_true_prob = np.zeros([2], dtype=np.float16)
        self.me_weights = np.zeros([self.num_weights], dtype=np.float16)
        self.other_weights = np.zeros([self.num_weights], dtype=np.float16)
        self.sender_index = np.zeros([self.num_past, 1], dtype=np.uint8)
        self.receiver_index = np.zeros([self.num_past, 1], dtype=np.uint8)
        # self.weights = weights
        # self.num_population = len(weights)
    def reduce_message(self, messages, send_me):
        # send_me : if -1, other send, else me send (200, 40)
        # output : concat(sum of me send, sum of other send)
        send_me[send_me != -1] = 1
        send_me[send_me == -1] = 0
        send_me = np.expand_dims(send_me,axis=-1)
        send_me = send_me.repeat(int(self.msg_shape / 2), axis=-1)
        recv_me = abs(send_me - 1)
        send_messages = send_me * messages
        recv_messages = recv_me * messages
        send_messages = send_messages.sum(1, keepdims=True).squeeze(1)
        recv_messages = recv_messages.sum(1, keepdims=True).squeeze(1)
        added = np.concatenate([send_messages, recv_messages], axis=-1) / 40 # max num message: 40
        return added.astype(np.float16)

    def extract(self, episodes, output_dir, proc_id):
        #nt 당 데이터 하나
        # past_current_idx = np.random.permutation(np.arange(self.args.num_same_population_episode))
        candidates = []
        for x in range(self.num_powers):
            for y in range(self.num_powers):
                if x == y :
                    continue
                candidates.append((x,y))

        mes = np.zeros((self.num_past + 1), dtype=np.uint8)
        others = np.zeros((self.num_past + 1), dtype=np.uint8)
        one_msg_shape = self.msg_shape - 2 # 88
        past_epi = 0
        for i in range(self.num_past + 1):
            # past episode
            try:
                with open(f"{args.proc_dir}/get_state/agent_{num_population}_id_{self.args.number}/"
                            f"{episodes}_{i}.npz", 'rb') as f:
                    npz = np.load(f)
                    me, other, epi_step = npz['me'], npz['other'], npz['epi_step']
                    other_map_tactician, other_internal_states = npz['other_map_tactician'], npz['other_internal_states']
                    me_map_tactician, me_internal_states = npz['me_map_tactician'], npz['me_internal_states']
                    board_states, orders_target, orders_src = npz['boards'], npz['orders_target'], npz['orders_src']
                    orders_dst, src_powers, dst_powers, orders_len = npz['orders_dst'], npz['src_powers'], npz['dst_powers'], npz['orders_len']
                    msg_list, msg_len, me_send = npz['messages'], npz['msg_len'], npz['msg_send_ind']
                    msgs_prob, orders_ind, orders_prob = npz['msgs_prob'], npz['orders_ind'], npz['orders_prob']
                    betray, fulfill = npz["betray"], npz["fulfill"]
            except:
                print(f"{args.proc_dir}/get_state/agent_{num_population}_id_{self.args.number}/"
                      f"{episodes}_{i}.npz")
                return None
            me_idx, me_weight = me[0], me[1:]
            #print(me_weight)
            other_idx, other_weight = other[0], other[1:]
            # make order
            order = np.concatenate([orders_target, orders_src, orders_dst, src_powers, dst_powers], axis=-1)
            order = np.sum(order, axis=1, dtype=np.float16) / self.max_action # max_action : 17
            if i == 0:
                # print(msg_len, past_current_idx[i])
                msg_nonzero_step = np.where(msg_len != 0)[0]
                if args.data_type == "normal":
                    first_cand_step = msg_nonzero_step
                elif args.data_type == "betray":
                    betray_step = np.where(np.any(betray == 1, axis=1))[0]
                    first_cand_step = np.intersect1d(msg_nonzero_step, betray_step)
                elif args.data_type == "fulfill":
                    fulfill_step = np.where(np.any(fulfill == 1, axis=1))[0]
                    first_cand_step = np.intersect1d(msg_nonzero_step, fulfill_step)
                else:
                    raise ValueError("ERROR, Check data type")

                chosen_step = np.random.choice(first_cand_step, 1)[0]  ## msg order 둘다 none 아닐때 // betray 일 때 // fulfill 일 때

                # choose src unit
                # ith_order = np.random.randint(0, orders_len[chosen_step], 1)
                jth_candidates = np.where(me_send[chosen_step] != -1)[0]
                jth_message = np.random.choice(jth_candidates, 1)[0]
                jth_order = np.where(orders_ind[chosen_step] == me_send[chosen_step][jth_message])[0][0]
                # print(orders_ind[chosen_step, jth_order], me_send[chosen_step][jth_message])
                self.input_src = orders_src[chosen_step, jth_order] #/ self.max_action
                #self.input_send = msg_list[chosen_step, jth_message][:one_msg_shape]
                self.input_send = msg_list[chosen_step, jth_message][:self.one_msg_shape] #/ 40
                self.target_dst = orders_dst[chosen_step, jth_order]
                self.target_true_prob = np.array([msgs_prob[chosen_step, jth_message],
                                                        orders_prob[chosen_step, jth_order]], dtype=np.float16)
                #recv = msg_list[chosen_step, jth_message][one_msg_shape:]
                recv = msg_list[chosen_step, jth_message][self.one_msg_shape:]
                self.target_order = orders_target[chosen_step, jth_order]
                self.target_recv = np.where(recv == 1)[0]

                # print('CHOSEN STEP. ORDER INDEX MSG INDEX', chosen_step, jth_order, jth_message)
                # print('MSG', msgs_prob[chosen_step])
                # print('ORDER', orders_prob[chosen_step])
                # print('INPUT SEND', self.input_send)
                # print('INPUT SRC', self.input_src)
                # print('ME OTHER INDEX', me, other)
                # print(f'TARGET DST {self.target_dst} ORDER {self.target_order} RECV {self.target_recv} TRUE {self.target_true_prob}')
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
                self.me_weights = me_weight
                self.other_weights = other_weight
                mes[-1] = me_idx
                others[-1] = other_idx
                curr_idx = past_epi
            else:
                self.past_order[past_epi] = order[:self.num_step]
                self.past_me_map_tactician[past_epi] = me_map_tactician[:self.num_step]
                self.past_me_internal_state[past_epi] = me_internal_states[:self.num_step]
                self.past_other_map_tactician[past_epi] = other_map_tactician[:self.num_step]
                self.past_other_internal_state[past_epi] = other_internal_states[:self.num_step]
                self.past_msg_state[past_epi] = self.reduce_message(msg_list[:self.num_step], me_send)
                self.past_board_state[past_epi] = board_states[:self.num_step]
                mes[past_epi] = me_idx
                others[past_epi] = other_idx
                past_epi = past_epi + 1
                if me_weight[0] != self.me_weights[0]:
                    print('MEWEIGHT DIFF', f"{args.proc_dir}/get_state/agent_{num_population}_id_{self.args.number}/"
                        f"{episodes}_{i}.npz", me_weight, self.me_weights)
                if other_weight[0] != self.other_weights[0]:
                    print('OTHWEIGHT DIFF', f"{args.proc_dir}/get_state/agent_{num_population}_id_{self.args.number}/"
                        f"{episodes}_{i}.npz", other_weight, self.other_weights)


        files = ['me_weights', 'other_weights', 'curr_me_map_tactician', 'curr_other_map_tactician', 'curr_order',
                 'curr_src', 'curr_send', 'curr_me_internal', 'curr_other_internal', 'curr_board', 'curr_message',
                 'past_me_map_tactician', 'past_other_map_tactician', 'past_order', 'past_me_internal', 'past_board',
                 'past_message', 'past_other_internal', 'other_index', 'me_index',
                 'target_order', 'target_dst', 'target_recv', 'target_true_prob']
        datas = [self.me_weights, self.other_weights, self.curr_me_map_tactician, self.curr_other_map_tactician, self.curr_order,
                 self.input_src, self.input_send, self.curr_me_internal_state, self.curr_other_internal_state, self.curr_board_state, self.curr_msg_state,
                 self.past_me_map_tactician, self.past_other_map_tactician, self.past_order, self.past_me_internal_state, self.past_board_state,
                 self.past_msg_state, self.past_other_internal_state, others, mes,
                 self.target_order, self.target_dst, self.target_recv, self.target_true_prob]
        for file, data in zip(files, datas):
            with open(f'{output_dir}/{file}_{episodes}.npz', 'wb') as f:
                np.savez_compressed(f, data=data)

    def concat(self, output_dir, final_dir):
        files = ['me_weights', 'other_weights', 'curr_me_map_tactician', 'curr_other_map_tactician', 'curr_order',
                 'curr_src', 'curr_send', 'curr_me_internal', 'curr_other_internal', 'curr_board', 'curr_message',
                 'past_me_map_tactician', 'past_other_map_tactician', 'past_order', 'past_me_internal', 'past_board',
                 'past_message', 'past_other_internal', 'other_index', 'me_index',
                 'target_order', 'target_dst', 'target_recv', 'target_true_prob']
        for file in tqdm(files, position=1, leave=True):
            rows = [row for row in os.listdir(output_dir) if row.startswith(file)]
            nrow = len(rows)
            with np.load(output_dir + '/' + rows[0], 'rb') as f:
                npz = f
                first = npz['data']
            if ('internal' in file) or ('weights' in file):
                total = np.zeros([nrow, *first.shape], dtype=np.float16)
            elif ('num_map_tact' in file):
                total = np.zeros([nrow, *first.shape], dtype=np.int32)
            else:
                total = np.zeros([nrow, *first.shape], dtype=np.uint8)
            for i, r in enumerate(rows):
                with np.load(output_dir + '/' + r, 'rb') as f:
                    ith_data = f
                    total[i] = ith_data['data']
                #os.remove(output_dir + '/' + r)
            np.savez_compressed(final_dir + '/' + file + '.npz', total=total)

def output(proc_id):
    # workloads = [input_list[args.num_cpu * i + proc_id] for i in range(len(input_list)//args.num_cpu + 1) if args.num_cpu * i + proc_id < len(input_list)]
    prev_list = []
    for file in tqdm(input_list, desc= f'#{proc_id}', position=0, leave=True):
        episodes = file[:file.find(".")+1][:file.find("[")-2]
        if episodes in prev_list:
            continue
        prev_list.append(episodes)
        preprocessor2.extract(episodes, output_dir, proc_id)
        #print(proc_id)

if __name__=="__main__":
    args = parse_args()

    preprocessed_dir = args.proc_dir
    num_population = args.num_population


    run_id = args.number
    output_dir = preprocessed_dir + f"/storage/agent_{num_population}_id_{run_id}"
    final_dir = preprocessed_dir + f"/final/agent_{num_population}_id_{run_id}"
    utils.make_dirs(output_dir)
    utils.make_dirs(final_dir)
    preprocessor2 = Storage(args)
    input_list = os.listdir(preprocessed_dir + f"/get_state/agent_{num_population}_id_{run_id}")
    input_list = sorted(input_list)
    print('------NUMBER {} start--------'.format(run_id), len(input_list))
    num_cpu = args.num_cpu #if args.num_cpu < 5 else 5
    try:
        output(0)
    except:
        logging.basicConfig(filename=f'./test{args.number}_concat.log', level=logging.ERROR)
        logging.error(traceback.format_exc())
    '''
    
    for file in tqdm(input_list, position=0, leave=True):
        episodes = file[:file.find(".")+1][:file.find("[")-2]
        preprocessor2.extract(episodes, output_dir)
    '''
    #print('-------CONCAT START {}-------'.format(run_id))
    #preprocessor2.concat(output_dir, final_dir)