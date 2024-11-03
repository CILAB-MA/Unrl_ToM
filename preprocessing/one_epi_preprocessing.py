import os, sys, yaml
sys.path.append(os.getcwd())
sys.path.append("/app/Dip_ToM/")
import argparse
import time
import random
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from preprocessing.word_preprocessing import MessageMaker
from preprocessing.board_preprocessing import BoardMaker
from preprocessing.log_to_np_utils import *
from utils.utils import *
import logging
import traceback

def parse_args():
    parser = argparse.ArgumentParser()
    # standard, small, ancmed, pure, clonial, empire, known_world_901, modern, world
    parser.add_argument('--env_type', '-et', type=str, default="standard")
    parser.add_argument('--num_same_population_episode', '-nsp', type=int, default=5)  # 총 nt * np * p  만큼의 에피소드가 생성됨
    parser.add_argument('--base_dir', '-b', type=str, default='./preprocessing/log_data')
    parser.add_argument('--proc_dir', '-pd', type=str, default='./preprocessing/preprocessed_data')
    parser.add_argument('--num_population', '-np', type=int, default=30)
    parser.add_argument('--number', '-n', type=int, default=0)
    parser.add_argument('--num_cpu', '-p', type=int, default=40)
    args = parser.parse_args()
    return args

def split_workload(log_list, num_cpu, proc_id):
    log_list = sorted(log_list)
    ext = log_list[0][-4:]
    epi_with_same_weights = sorted(set([log[:-5] for log in log_list if log[:3]]))

    n = len(epi_with_same_weights)
    epi_splitted = sorted(set([epi_with_same_weights[num_cpu * i + proc_id] for i in range((n//num_cpu)) if num_cpu * i + proc_id < n]))
    workloads = []
    for epi in epi_splitted:
        for i in range(args.num_same_population_episode):
            workloads.append(epi + str(i) + ext)
    return workloads

def output(proc_id):
    # dirs and num_cpu
    preprocessed_dir = args.proc_dir
    run_id = args.number
    base_dir = args.base_dir
    config_path = "./configs/{}.yaml".format(args.env_type)

    # seed = (os.getpid() * int(time.time())) % 1234
    # np.random.seed(seed)
    # random.seed(seed)

    with open(config_path) as f:
        yd = yaml.load(f, Loader=yaml.FullLoader)

    # about diplomacy features
    num_population = args.num_population
    num_agent = yd['num_agent']
    num_loc = yd['num_loc']
    max_action = yd['max_action']
    num_step = yd['num_step']

    # input data used both past and curr
    internal_shape = (3,)
    map_tactician_shape = (yd['num_map_tact'],)
    msg_shape = (40, 5 * num_agent + 10)
    board_shape = (num_loc, num_agent * 3 + 14)

    # input data only used in curr
    input_src_shape = (max_action, num_loc)

    # target shape
    target_order_shape = (max_action, 4)
    target_dst_shape = (max_action, num_loc)
    dst_power_shape = (max_action, num_agent + 1)
    src_power_shape = (max_action, num_agent)

    powers = sorted(yd['powers'])

    output_dir = preprocessed_dir + f"/get_state/agent_{num_population}_id_{run_id}"
    make_dirs(output_dir)
    log_list = os.listdir(base_dir + f"/agent_{num_population}_id_{run_id}")
    workloads = split_workload(log_list, args.num_cpu, proc_id)
    print('----------PROC {} WORKLOAD START WITH NUMBER {}, WORKLOAD '.format(proc_id, run_id))
    for file in tqdm(workloads , position=0, leave=True):
        filepath = f'{base_dir + f"/agent_{num_population}_id_{run_id}"}/{file}'
        with open(filepath, 'rb') as f:
            npz = np.load(f, allow_pickle=True)
            sampled_weight, shuffled_idx, log = npz['sampled_weights'], npz['shuffle'], npz['epi_log'].item()
        # input data used both past and curr
        other_internal_states = np.zeros([num_step, internal_shape[0]], dtype=np.float16)
        other_map_tactician = np.zeros([num_step, map_tactician_shape[0]], dtype=np.float16)
        me_internal_states = np.zeros([num_step, internal_shape[0]], dtype=np.float16)
        me_map_tactician = np.zeros([num_step, map_tactician_shape[0]], dtype=np.float16)
        messages = np.zeros([num_step, msg_shape[0], msg_shape[1]], dtype=np.uint8)
        boards = np.zeros([num_step, board_shape[0], board_shape[1]], dtype=np.uint8)
        # input data only used in curr
        orders_src = np.zeros([num_step, input_src_shape[0], input_src_shape[1]], dtype=np.uint8)
        orders_ind = np.full([num_step, input_src_shape[0], 10], -1, dtype=np.int8)
        orders_prob = np.zeros([num_step, input_src_shape[0]], dtype=np.float16)
        msg_len = np.zeros([num_step, 1], dtype=np.int8)
        msg_send_ind = np.zeros([num_step, 40], dtype=np.int8)
        msgs_prob = np.zeros([num_step, 40], dtype=np.float16)
        betray_info = np.zeros([num_step, input_src_shape[0]], dtype=np.float16)
        fulfill_info = np.zeros([num_step, input_src_shape[0]], dtype=np.float16)
        # target shape
        orders_target = np.zeros([num_step, target_order_shape[0], target_order_shape[1]], dtype=np.uint8)
        orders_dst = np.zeros([num_step, target_dst_shape[0], target_dst_shape[1]], dtype=np.uint8)

        # etc
        src_powers = np.zeros([num_step, src_power_shape[0], src_power_shape[1]], dtype=np.uint8)
        dst_powers = np.zeros([num_step, dst_power_shape[0], dst_power_shape[1]], dtype=np.uint8)
        orders_len = np.zeros([num_step, 1], dtype=np.uint8)

        i = -1
        n = -1
        if file[-5] == '0':
            # 0 -> current episode
            other_candidates = [power for power, num_msg in log['num_messages_epi'].items() if sum(num_msg.values()) > 0] # choose me who interact msg with others
            other_power = random.sample(other_candidates, 1)[0]
            other_power_idx = powers.index(other_power)
            other_idx = shuffled_idx[other_power_idx]
            me_candidates = [power for power, num_msg in log['num_messages_epi'][other_power].items() if num_msg > 0] # choose other who interact msg with me
            # weights = [- num_msg for num_msg in log['num_messages_epi'][other_power].values() if num_msg > 0] # high weights for low interaction number
            # weights = np.array(weights)
            # weights = np.exp(weights - np.max(weights))
            # weights = weights / weights.sum()
            # me_power = random.choices(me_candidates, weights=weights)[0]
            me_power = random.sample(me_candidates, 1)[0]
            me_power_idx = powers.index(me_power)
            me_idx = shuffled_idx[me_power_idx]
            file_split_0 = file.split('_')
            epi_num_0 = file_split_0[1]
        file_split = file.split('_')
        epi_num = file_split[1]
        if epi_num_0 != epi_num:
            print('WRONG!!!!!!!!!!!!!!!!!!!!!!', file_split, file_split_0)
        # (power idx, power weight)
        me = (np.where(shuffled_idx == me_idx)[0][0], powers[np.where(shuffled_idx == me_idx)[0][0]], sampled_weight[me_idx])
        other = (np.where(shuffled_idx == other_idx)[0][0], powers[np.where(shuffled_idx == other_idx)[0][0]], sampled_weight[other_idx])
        call_static = True
        # print(sampled_weight, file, log['num_messages_epi'][other[1]],  log['num_messages_epi'][other[1]][me[1]])
        while True:
            i += 1
            if call_static:
                message_maker = MessageMaker(num_agent, powers, log['static_infos'])
                board_maker = BoardMaker(num_agent, powers, log['static_infos'])
                call_static = False
            if i == len(log['infos']):
                break
            if (log['infos'][i]['name'][-1] != 'M'):
                continue
            n += 1
            # print(f'---------------{n} phase-------------')
            if n == num_step:
                break
            locs = log["static_infos"]["locs"]
            log["static_infos"]["powers"] = powers
            messages[n], msg_len[n], msg_send_ind[n], msgs_prob[n] = message_maker.make_message(
                log['messages'][i],
                me[1], other[1],
                log['infos'][i]['loc2power'])

            boards[n] = board_maker.make_state(log, i)
            other_map_tactician[n] = maptactician_maker(log, i , other)
            other_internal_states[n] = internal_state_maker(log, i, me[1], other[1], powers, is_me=False)
            me_map_tactician[n] = maptactician_maker(log, i , me)
            me_internal_states[n] = internal_state_maker(log, i, me[1], other[1], powers, is_me=True)
            orders = order_maker(log, i, other[1], locs, powers, max_action, num_loc)
            orders_target[n], orders_src[n], orders_dst[n], src_powers[n], dst_powers[n], orders_len[n], orders_ind[n], orders_prob[n], betray_info[n], fulfill_info[n] = orders

            # print(orders_ind[n], msg_send_ind[n])
            # print('-------------------------------')
        if np.sum(msg_len) == 0 and file[-5] == '0':
            print(f'FILE IS {filepath}')
            print(f'ME: {powers[me[0]]}-{me} OTHER: {powers[other[0]]}-{other}')
            print(log['num_messages_epi'][me[1]])
        #     print(f'INTERACTED MSGS are {log["num_messages_epi"]}')

        epi_step = n
        me = np.concatenate((np.where(shuffled_idx == me_idx)[0], sampled_weight[me_idx]))
        other = np.concatenate((np.where(shuffled_idx == other_idx)[0], sampled_weight[other_idx]))
        probs = [(sampled_weight[other_idx][1] + 1) / 2, 1 - (sampled_weight[other_idx][1] + 1) / 2,
                 (sampled_weight[me_idx][1] + 1) / 2, 1 - (sampled_weight[me_idx][1] + 1) / 2, 0]

        probs = np.array(probs)
        avail_prob = np.in1d(np.unique(msgs_prob), probs)
        # print(False in avail_prob, avail_prob)
        # pseudo_array = np.array([False, True, True])
        # print(False in pseudo_array)
        if False in avail_prob:
            print(avail_prob, np.unique(msgs_prob), probs)
        if len(np.unique(msgs_prob)) > 5:
            print(sampled_weight[me_idx], sampled_weight[other_idx], np.unique(msgs_prob), sampled_weight)
        # # print(me, other)
        with open(f"{output_dir}/{file[:-4]}.npz", 'wb') as f:
            np.savez_compressed(f, me=me, other=other, epi_step=epi_step, other_map_tactician=other_map_tactician,
                                other_internal_states=other_internal_states, me_map_tactician=me_map_tactician,
                                me_internal_states=me_internal_states, boards=boards, orders_target=orders_target,
                                betray=betray_info, fulfill=fulfill_info,
                                orders_src=orders_src, orders_dst=orders_dst, src_powers=src_powers, dst_powers=dst_powers,
                                orders_len=orders_len, messages=messages, msg_len=msg_len, msg_send_ind=msg_send_ind,
                                msgs_prob=msgs_prob, orders_prob=orders_prob, orders_ind=orders_ind)

if __name__ == '__main__':
    start = time.perf_counter()
    args = parse_args()
    num_cpu = args.num_cpu
    try:
        with mp.Pool(processes=num_cpu) as p:
            p.map(output, range(num_cpu))
    except:
        logging.basicConfig(filename=f'./test{args.number}_1epi.log', level=logging.ERROR)
        logging.error(traceback.format_exc())
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    print(f"Elapsed time : {time.perf_counter() - start}")

