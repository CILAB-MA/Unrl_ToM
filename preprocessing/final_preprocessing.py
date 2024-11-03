import numpy as np
from tqdm import tqdm
import os, sys, shutil
sys.path.append(os.getcwd())
from utils import utils
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # standard, small, ancmed, pure, clonial, empire, known_world_901, modern, world
    parser.add_argument('--num_population', '-np', type=int, default=30)
    parser.add_argument('--proc_dir', '-pd', type=str, default='./preprocessing/preprocessed_data')
    parser.add_argument('--number', '-n', type=int, default=0)
    args = parser.parse_args()
    return args

def concat(output_dir, final_dir):
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
        if ('internal' in file) or ('weights' in file) or ('true_prob' in file) or ('message' in file) or ('send' in file) or ('order' in file):
            total = np.zeros([nrow, *first.shape], dtype=np.float16)
        elif ('num_map_tact' in file):
            total = np.zeros([nrow, *first.shape], dtype=np.int32)
        else:
            total = np.zeros([nrow, *first.shape], dtype=np.uint8)
        for i, r in enumerate(rows):
            with np.load(output_dir + '/' + r, 'rb') as f:
                ith_data = f
                total[i] = ith_data['data']
            # os.remove(output_dir + '/' + r)
        np.savez_compressed(final_dir + '/' + file + '.npz', total=total)

if __name__=="__main__":
    args = parse_args()

    preprocessed_dir = args.proc_dir
    num_population = args.num_population
    run_id = args.number
    output_dir = preprocessed_dir + f"/storage/agent_{num_population}_id_{run_id}"
    final_dir = preprocessed_dir + f"/final/agent_{num_population}_id_{run_id}"
    utils.make_dirs(output_dir)
    utils.make_dirs(final_dir)
    print('-------CONCAT START {}-------'.format(run_id))
    concat(output_dir, final_dir)
    # shutil.rmtree(output_dir)