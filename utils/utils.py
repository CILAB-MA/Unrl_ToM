import numpy as np
import os
import pickle
import torch as tr
from math import pi

def generate_weights(num_agents, num_pool, num_adviser=4):
    file_name = f'population_pool_{num_pool}.pickle'
    weights = np.random.rand(num_agents, num_adviser)

    with open(file_name, 'wb') as f:
        pickle.dump(weights, f)


def load_weights(file_name):
    with open(file_name, 'rb') as f:
        population_weights = pickle.load(f)
    return population_weights


def make_weights(num_agent):
    population_weights = []
    for i in range(num_agent):
        weight_rad = 2 * pi / num_agent * i
        population_weights.append(weight_rad)

    return population_weights


def save_log_data(sampled_weights, shuffle, epi_log, base_dir, name, eval=None):
    make_dirs(base_dir)
    with open(base_dir + name, "wb") as f:
        np.savez_compressed(f, sampled_weights=sampled_weights, shuffle=shuffle, epi_log=epi_log)


def save_data(npy_data, is_train, base_dir, eval=None):
    keys = npy_data.keys()

    folder_name = os.path.join(base_dir, is_train)
    if eval != None:
        folder_name = os.path.join(folder_name, str(eval))
    make_dirs(folder_name)

    for key in keys:
        np.save(folder_name + '/' + key, npy_data[key])


def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


if __name__ == '__main__':

    generate_weights()
