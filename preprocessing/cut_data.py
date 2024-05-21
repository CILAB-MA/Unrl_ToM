import numpy as np
import os
BASE_DIR = '/data/storage_shorten'
SAVE_DIR = '/data/cut_shorten/'

touint8 = ['past_order', 'past_message', 'past_board',
           'curr_message', 'curr_board', 'curr_order', 'curr_src', 'curr_send',
           'target_recv', 'target_order', 'target_dst',
           'me_index', 'other_index']
def cut_data():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    for agent_id in os.listdir(BASE_DIR):
        folder = os.path.join(BASE_DIR, agent_id)
        if '12' not in agent_id:
            continue
        new_folder = os.path.join(SAVE_DIR, agent_id)
        #if not os.path.exists(new_folder):
        #    os.makedirs(new_folder)
        #curr_msg = np.load(os.path.join(folder, 'curr_message.npy'))
        #msg_sum = curr_msg.sum(-1).sum(-1).sum(-1)


        for npy_file in os.listdir(folder):
            npy = np.load(os.path.join(folder, npy_file))
            print('BEFORE', SAVE_DIR, npy_file, npy.shape)
            print(npy[5000:].sum())
            '''
            if npy_file[:-4] in touint8:    
                npy = npy.astype(np.uint8)
            if 'send' in npy_file:
                newf_file = os.path.join(new_folder, npy_file)
                np.save(newf_file, npy)
                continue
            elif 'src' in npy_file:
                newf_file = os.path.join(new_folder, npy_file)
                np.save(newf_file, npy)
                continue
            if 'past' in npy_file:
                cut_step = 100
            elif 'curr' in npy_file:
                cut_step = 20
            else:
                newf_file = os.path.join(new_folder, npy_file)
                np.save(newf_file, npy)
                continue
            if 'curr' in npy_file:
                if 'order' in npy_file:
                    npy = npy[:, -cut_step +1 :]
                else:
                    npy = npy[:, -cut_step:]
            else:
                npy = npy[:, :, :cut_step]
            new_file = os.path.join(new_folder, npy_file)
            #np.save(new_file, npy)
            print('AFTER', npy_file, npy.shape)
            '''



if __name__ == '__main__':
    cut_data()