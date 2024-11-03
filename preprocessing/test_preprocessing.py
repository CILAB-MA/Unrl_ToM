import numpy as np
import os

def main():
    base_dir = '/data/map_test_hc/preprocess/clonial/final/agent_30_id_1/'
    for file in os.listdir(base_dir):
        f = np.load(os.path.join(base_dir, file) , 'rb')
        print(file, f['total'].shape)

if __name__ == '__main__':
    main()