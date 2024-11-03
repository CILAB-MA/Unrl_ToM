import argparse
import yaml, os
from experiments.tools import make_folder
import wandb
import torch
import random
import numpy as np
import importlib
import time
from experiments import MODULES

def parse_args():
    parser = argparse.ArgumentParser('For ToM Passive Exp')
    parser.add_argument('--map_type', '-m', default="standard")
    parser.add_argument('--exp', default='only_lstm')
    parser.add_argument('--seed', type=int, default='0')
    parser.add_argument('--data_dir', '-d', type=str, default='/app/nas_data')
    parser.add_argument('--charnet_type', '-c_type', type=str, default='fc')
    parser.add_argument('--prednet_type', '-p_type', type=str, default='only_lstm')

    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--no_char', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gcn', action='store_true', default=False)
    return parser.parse_args()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_module_func(module_name, pkg):
    print(pkg)
    mod = importlib.import_module(module_name)
    return mod


def main(configs, exp):
    if configs['use_wandb']:
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
        wandb.init(project=configs['project'], entity=configs['entity'],
                   name='Final! [{}] char: {{{}}}, pred: {{{}}}, no_char: {{{}}}, lr: {{{}}}, map_type: {{{}}}'
                   .format(configs['exp'], configs['charnet_type'], configs['prednet_type'], configs['no_char'], configs['lr'], configs['map_type']))
        wandb.config.update(configs)

    start = time.time()
    print('START Experiment {} with seed {} => model save: {}, use wandb: {}, map_type: {}'.format(configs['exp'], configs['seed'], configs['save_model'], configs['use_wandb'], configs['map_type']))
    print("Start time: ", time.ctime())
    print('num_data_per_loader : ', configs['num_data_per_loader'])

    experiment_folder = None
    if configs['save_model']:
        experiment_folder = make_folder(configs['exp'])

    exp.run_experiment(exp_dir=experiment_folder, configs=configs)
    print("This experiment took {} seconds".format(round(time.time() - start, 1)))


if __name__ == '__main__':
    args = parse_args()

    exp_name = args.exp
    map_name = args.map_type
    pwd = os.getcwd()[1:]
    exp = MODULES[exp_name]
    #exp_module = load_module_func(".experiments.{}".format(exp_name), pwd)
    set_seed_everywhere(args.seed)

    with open("./experiments/config.yaml") as f:
        all_configs = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join("./configs/", map_name + '.yaml')) as f:
        map_configs = yaml.load(f, Loader=yaml.FullLoader)

    configs = dict(all_configs["basic"], **all_configs["{}".format(exp_name)])
    configs.update(map_configs)

    for k, v in vars(args).items():
        configs[k] = v

    main(configs, exp)