import numpy as np
import os

def main():
    base_dir = '/data/exp_total/storage/agent_30_id_99'
    files = os.listdir(base_dir)
    for file in files:
        npy = os.path.join(base_dir, file)
        f = np.load(npy)
        print(file, f.shape, f.sum())
    pass

if __name__ == '__main__':
    main()