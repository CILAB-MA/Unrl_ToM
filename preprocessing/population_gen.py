import numpy as np
import pickle


def generate_weights(num_agents, num_pool, num_adviser=2):
    file_name = f'population_pool_{num_pool}.pickle'
    map_weights = np.random.rand(num_pool, 1)
    relation_weights = 1 - map_weights
    weights = np.column_stack([map_weights, relation_weights])

    with open(file_name, 'wb') as f:
        pickle.dump(weights, f)


if __name__ == '__main__':
    # make 30 different agents(weights) in standard game
    generate_weights(7, 30)
