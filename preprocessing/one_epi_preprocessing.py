import os, sys, copy
sys.path.append(os.getcwd())
import argparse
import time
import random
import multiprocessing as mp
import pickle as pkl
from tqdm import tqdm
from preprocessing.word_preprocessing import MessageMaker
from preprocessing.board_preprocessing import BoardMaker
from preprocessing.log_to_np_utils import *
from utils.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_type', '-et', type=int, default=1)  # 1:  기본 env, 2: small env
    parser.add_argument('--num_same_population_episode', '-nsp', type=int, default=5)  # 총 nt * np * p  만큼의 에피소드가 생성됨
    parser.add_argument('--base_dir', '-b', type=str, default='')
    parser.add_argument('--proc_dir', '-pd', type=str, default='')
    parser.add_argument('--num_population', '-np', type=int, default=30)
    parser.add_argument('--number', '-n', type=int, default=0)
    parser.add_argument('--num_step', '-ns', type=int, default=200)
    parser.add_argument('--num_cpu', '-p', type=int, default=40)
    args = parser.parse_args()
    return args

def split_workload(log_list, num_cpu, proc_id):
    log_list = sorted(log_list)
    ext = log_list[0][-7:]
    epi_with_same_weights = sorted(set([log[:-8] for log in log_list if log[:6]]))

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

    # about diplomacy features
    num_population = args.num_population
    num_agent = 7 if args.env_type == 1 else 3
    num_loc = 81 if args.env_type == 1 else 28
    max_action = 17 if args.env_type == 1 else 6 # half of the sc
    num_step = 200 if args.env_type == 1 else 100

    # input data used both past and curr
    internal_shape = (3,)
    map_tactician_shape = (120,)
    msg_shape = (40, 5 * num_agent + 10)
    board_shape = (num_loc, num_agent * 3 + 14)

    # input data only used in curr
    input_src_shape = (max_action, num_loc)

    # target shape
    target_order_shape = (max_action, 4)
    target_dst_shape = (max_action, num_loc)
    dst_power_shape = (max_action, num_agent + 1)
    src_power_shape = (max_action, num_agent)


    if args.env_type == 1:
        powers = ['FRANCE', 'AUSTRIA', 'GERMANY', 'RUSSIA', 'ITALY', 'TURKEY', 'ENGLAND']
    elif args.env_type == 2:
        powers = ['FRANCE', 'GERMANY', 'ITALY']

    output_dir = preprocessed_dir + f"/get_state/agent_{num_population}_id_{run_id}"
    make_dirs(output_dir)
    log_list = os.listdir(base_dir + f"/agent_{num_population}_id_{run_id}")
    workloads = split_workload(log_list, args.num_cpu, proc_id)

    for file in tqdm(workloads):
        filepath = f'{base_dir + f"/agent_{num_population}_id_{run_id}"}/{file}'
        with open(filepath, 'rb') as f:
            (sampled_weight, shuffled_idx), log = pkl.load(f)
        # input data used both past and curr
        other_internal_states = np.zeros([num_step, internal_shape[0]], dtype=np.float16)
        other_map_tactician = np.zeros([num_step, map_tactician_shape[0]], dtype=np.float16)
        me_internal_states = np.zeros([num_step, internal_shape[0]], dtype=np.float16)
        me_map_tactician = np.zeros([num_step, map_tactician_shape[0]], dtype=np.float16)
        messages = np.zeros([num_step, msg_shape[0], msg_shape[1]], dtype=np.uint8)
        boards = np.zeros([num_step, board_shape[0], board_shape[1]], dtype=np.uint8)

        # input data only used in curr
        orders_src = np.zeros([num_step, input_src_shape[0], input_src_shape[1]], dtype=np.uint8)
        msg_len = np.zeros([num_step, 1], dtype=np.uint8)
        msg_send_ind = np.zeros([num_step, 40], dtype=np.uint8)

        # target shape
        orders_target = np.zeros([num_step, target_order_shape[0], target_order_shape[1]], dtype=np.uint8)
        orders_dst = np.zeros([num_step, target_dst_shape[0], target_dst_shape[1]], dtype=np.uint8)

        # etc
        src_powers = np.zeros([num_step, src_power_shape[0], src_power_shape[1]], dtype=np.uint8)
        dst_powers = np.zeros([num_step, dst_power_shape[0], dst_power_shape[1]], dtype=np.uint8)
        orders_len = np.zeros([num_step, 1], dtype=np.uint8)

        i = -1
        n = -1
        if file[-8] == '0':
            # select index of sampled_weight
            me_idx, other_idx = random.sample(list(range(len(powers))), 2)

        # (power idx, power name, power weight)
        me = (np.where(shuffled_idx == me_idx)[0][0], powers[np.where(shuffled_idx == me_idx)[0][0]], sampled_weight[me_idx])
        other = (np.where(shuffled_idx == other_idx)[0][0], powers[np.where(shuffled_idx == other_idx)[0][0]], sampled_weight[other_idx])
        call_static = True
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
            if n == 200:
                break
            locs = log["static_infos"]["locs"]
            log["static_infos"]["powers"] = powers
            messages[n], msg_len[n], msg_send_ind[n] = message_maker.make_message(log['messages'][i], me[0], other[0],
                                                           log['infos'][i]['loc2power'])
            boards[n] = board_maker.make_state(log, i)
            other_map_tactician[n] = maptactician_maker(log, i , other)
            other_internal_states[n] = internal_state_maker(log, i, me[0], other[0], powers, is_me=False)
            me_map_tactician[n] = maptactician_maker(log, i , me)
            me_internal_states[n] = internal_state_maker(log, i, me[0], other[0], powers, is_me=True)

            orders = order_maker(log, i, other[0], locs, powers)
            orders_target[n], orders_src[n], orders_dst[n], src_powers[n], dst_powers[n], orders_len[n] = orders
        epi_step = n
        with open(f"{output_dir}/{file[:-7]}.pickle", 'wb') as f:
            pkl.dump(((me, other, epi_step),
                      other_map_tactician, other_internal_states,
                      me_map_tactician, me_internal_states, boards,
                      orders_target, orders_src, orders_dst, src_powers, dst_powers, orders_len), f)

        with open(f"{output_dir}/{file[:-7]}[{me[0]}_to_{other[0]}].pickle", 'wb') as f:
            pkl.dump((messages, msg_len, msg_send_ind), f)

if __name__ == '__main__':
    start = time.perf_counter()
    args = parse_args()
    num_cpu = args.num_cpu
    try:
        with mp.Pool(processes = num_cpu) as p:
            p.map(output, range(num_cpu))
    except:
        from multiprocessing.pool import ThreadPool
        with ThreadPool(num_cpu) as p:
            p.map(output, range(num_cpu))
    print(f"Elapsed time : {time.perf_counter() - start}")
    
