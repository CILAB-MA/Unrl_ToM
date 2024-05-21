import numpy as np
import pickle as pkl
import os

class DataEvaluation:

    def __init__(self, args):

        self.base_dir = args.base_dir

    def read_pkl(self):
        folder = 'agent_{}_id_{}'.format(self.num_population, self.run_id)
        log_path = os.path.join(os.path.join(self.base_dir, folder))
        for log in os.listdir(log_path):
            with open(os.path.join(log_path, log), 'rb') as f:
                (sampled_weight, shuffled_ind), log = pkl.load(f)



