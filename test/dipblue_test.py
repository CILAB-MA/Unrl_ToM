import gym, logging, os, sys

from gym.envs.registration import register
import numpy as np
sys.path.append(os.getcwd())

from dipbluebot.dipblue import DipBlue
import matplotlib.pyplot as plt
import argparse

def make_one_weight(num_agent, feature_dim,others_constant=1):
    props = np.full((num_agent, 2), others_constant).astype(np.float16)
    feat = np.random.normal(0.5, 0.25, num_agent)
    props[:, feature_dim] = feat
    return props

def visualize(sample_weights, trajs, names, feature_dim, c=[0.2, 0.5], mode='one'):
    '''
    weights : adviser weight
    trajs : data (sum of msg or var of region)
    names : [$WHAT ADVISER, $WHAT DATA]
    x-axis : phase
    y-axis : number
    color : weights
    '''
    plt.clf()
    plt.figure(figsize=(30, 5))

    mu_per_phase = np.zeros(1000)
    mu_number = np.zeros(1000)
    max_num = 0
    for weight, traj in zip(sample_weights, trajs):
        if mode == 'one':
            weight_var = abs(weight)
            weight_var = weight_var[0]
            label = 'weight {}'.format(round(weight_var, 2))
        elif mode == 'all':
            weight_var = np.var(weight)
            label='var {}'.format(round(weight_var, 2))
        else:
            weight_var = np.var(weight[:, feature_dim])
            label='var {}'.format(round(weight_var, 2))
        x = np.arange(len(traj))
        mu_per_phase[:len(traj)] += traj
        mu_number[:len(traj)] += 1
        if weight_var / 3 > 1:
            weight_var = 3
        plt.plot(x, np.reshape(np.array(traj), (-1, 1)),
                 color=(round(weight_var, 2)/3, c[0], c[1]), label=label, alpha=0.5)
        if len(traj) > max_num:
            max_num = len(traj)
    mu_per_phase = mu_per_phase[:max_num] / mu_number[:max_num]
    plt.plot(np.arange(max_num), mu_per_phase, marker='o',  mfc='black', markersize=4)
    plt.xlabel('Phase')
    plt.ylabel(names[1])
    plt.legend()
    plt.savefig('{}_{}.png'.format(names[0], names[1]), dpi=100)

def dipblue_tester_one_country(num_sampling=1000, feature_dim=0, num_player=7):
    features = ['SITUATION', 'RELATION']
    register(
        id='PressDiplomacyEnv-v0',
        entry_point='environment.env:PressDiplomacyEnv'
    )
    env = gym.make('PressDiplomacyEnv-v0')
    env.game_type = 'standard'
    env.reset()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info('CHECK FOR SAMPLED WEIGHTS....')

    #logger.info('SAMPLED WEIGHTS : {}'.format(sampled_weights))

    powers = ['FRANCE', 'AUSTRIA', 'GERMANY', 'RUSSIA', 'ITALY', 'TURKEY', 'ENGLAND']
    sampled_weights = np.random.rand(num_sampling, 1)

    for me_power in range(7):
        total_weights = []
        total_messages = []
        total_regions = []

        for ith_game in range(num_sampling):
            num_messages = []
            num_regions = []
            weights = np.zeros((7, 2)) + 0.01
            weights[:, feature_dim] = 0.99
            weights[me_power, 0] = sampled_weights[ith_game]
            weights[me_power, 1] = 1 - sampled_weights[ith_game]
            logger.info('SAMPLED WEIGHTS : {}'.format(weights))
            players = {power: DipBlue(env.static_infos, power, weights=weight) for power, weight in
                       zip(list(env.static_infos['powers']), weights)}

            obs, infos = env.reset()
            total_weights.append(sampled_weights[ith_game])
            while env.is_done() is False:
                if not env.is_nego:
                    for power in powers:
                        power_orders, _, prev_order_clear = players[power].act(infos)

                        env.submit((power, power_orders), prev_order_clear)
                    obs, rew, dones, infos = env.step(None)

                else:
                    for power in powers:
                        negos, agreed_orders, _ = players[power].act(infos)
                        env.submit(negos, agreed_orders)
                    obs, rew, done, infos = env.step(None)

                msg_in_phase = [len(message) for message in infos['messages'].values()]
                region_in_phase = [len(loc) for loc in infos['power2loc'].values()]
                num_messages.append(np.sum(msg_in_phase))
                num_regions.append(np.var(region_in_phase))

            total_messages.append(num_messages)
            total_regions.append(num_regions)

            logger.info('{}TH GAME ENDED....'.format(ith_game))
        visualize(total_weights, total_messages,
                  names=['{}_{}'.format(features[feature_dim], powers[me_power]), 'NUM OF MSG'],
                  feature_dim=feature_dim)
        visualize(total_weights, total_regions,
                  names=['{}_{}'.format(features[feature_dim], powers[me_power]), 'VAR OF LOC'],
                  feature_dim=feature_dim)

def dipblue_tester_one_feat(num_sampling=1000, feature_dim=0, num_player=7):
    register(
        id='PressDiplomacyEnv-v0',
        entry_point='environment.env:PressDiplomacyEnv'
    )
    features = ['SITUATION', 'RELATION']
    env = gym.make('PressDiplomacyEnv-v0')
    env.game_type = 'standard'
    env.reset()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info('CHECK FOR SAMPLED WEIGHTS....')

    #logger.info('SAMPLED WEIGHTpS : {}'.format(sampled_weights))

    powers = ['FRANCE', 'AUSTRIA', 'GERMANY', 'RUSSIA', 'ITALY', 'TURKEY', 'ENGLAND']



    total_weights = []
    total_messages = []
    total_regions = []

    for ith_game in range(num_sampling):
        num_messages = []
        num_regions = []

        map_weights = np.random.rand(num_player, 1)
        relation_weights = 1 - map_weights
        weights = np.column_stack([map_weights, relation_weights])
        logger.info('SAMPLED WEIGHTS : {}'.format(weights))
        players = {power: DipBlue(env.static_infos, power, weights=weight) for power, weight in
                  zip(list(env.static_infos['powers']), weights)}

        obs, infos = env.reset()
        total_weights.append(weights)
        while env.is_done() is False:
            if not env.is_nego:
                for power in powers:
                    power_orders, _, prev_order_clear = players[power].act(infos)

                    env.submit((power, power_orders), prev_order_clear)
                obs, rew, dones, infos = env.step(None)

            else:
                for power in powers:
                    negos, agreed_orders, _ = players[power].act(infos)
                    env.submit(negos, agreed_orders)
                obs, rew, done, infos = env.step(None)

            msg_in_phase = [len(message) for message in infos['messages'].values()]
            region_in_phase = [len(loc) for loc in infos['power2loc'].values()]
            num_messages.append(np.sum(msg_in_phase))
            num_regions.append(np.var(region_in_phase))

        total_messages.append(num_messages)
        total_regions.append(num_regions)


        logger.info('{}TH GAME ENDED....'.format(ith_game))


    visualize(total_weights, total_messages, names=[features[feature_dim], 'NUM OF MSG'], feature_dim=feature_dim, mode='all')
    visualize(total_weights, total_regions, names=[features[feature_dim], 'VAR OF LOC'], feature_dim=feature_dim, mode='all')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-sampling', default=30, type=int)
    parser.add_argument('--feature-dim', default=0, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    #dipblue_tester_one_feat(args.num_sampling, args.feature_dim, num_player=7)
    dipblue_tester_one_country(args.num_sampling, args.feature_dim)

