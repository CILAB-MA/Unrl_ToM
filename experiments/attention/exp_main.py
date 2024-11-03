import wandb
import copy
from torch.utils.data import DataLoader

from experiments import dataset
from experiments.tools import *
from utils.visualize import Visualizer
from .PredNet import *
import torch
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_one_loader(train_loader, tom_net, criterion, optimizer, device, no_char=False):
    train_epoch_order_acc = 0
    train_epoch_dst_acc = 0
    train_epoch_response_acc = 0
    train_epoch_loss = 0
    train_e_chars_list = []
    train_me_weights_list = []
    train_other_weights_list = []
    train_other_powers_list = []
    train_epoch_true_order_prob_loss = 0
    train_epoch_true_response_prob_loss = 0

    for i, batch in enumerate(train_loader):
        p_board, p_order, p_message, c_board, c_src, c_message, t_order, t_dst, t_response, \
                me_weights, other_weights, me_ind, other_ind, c_send, c_order, t_true_prob = batch

        p_board = p_board.float().to(device)
        p_order = p_order.float().to(device)
        p_message = p_message.float().to(device)
        c_board = c_board.float().to(device)
        c_src = c_src.float().to(device)
        c_message = c_message.float().to(device)
        c_send = c_send.float().to(device)
        c_order = c_order.float().to(device)
        t_order = tr.flatten(t_order.long().to(device))
        t_dst = tr.flatten(t_dst.long().to(device))
        t_response = tr.flatten(t_response.long().to(device))
        me_ind = me_ind.float().to(device)
        other_ind = other_ind.float().to(device)
        t_true_prob = t_true_prob.to(device)
        pred_order, pred_dst, pred_response, e_char, weights, e_char_raw = tom_net(p_board, p_order, p_message, c_board,
                                                                                   c_src, c_message, c_send, c_order,
                                                                                   other_ind, me_ind)
        order_loss = criterion(pred_order, t_order)
        dst_loss = criterion(pred_dst, t_dst)
        response_loss = criterion(pred_response, t_response)

        loss = order_loss + dst_loss + response_loss
        if not no_char:
            train_e_chars_list.append(e_char.cpu().detach().numpy())
            train_me_weights_list.append(me_weights.cpu().detach().squeeze(-1).numpy())
            train_other_weights_list.append(other_weights.cpu().detach().squeeze(-1).numpy())
            train_other_powers_list.append(other_ind[:, -1].cpu().detach().numpy())

        pred_order_inds = tr.argmax(pred_order, dim=-1)
        pred_dst_inds = tr.argmax(pred_dst, dim=-1)
        pred_response_inds = tr.argmax(pred_response, dim=-1)

        order_acc = tr.sum(pred_order_inds == t_order).item()
        dst_acc = tr.sum(pred_dst_inds == t_dst).item()
        response_acc = tr.sum(pred_response_inds == t_response).item()

        pred_response_prob = torch.exp(pred_response[torch.arange(pred_response.size(0)), t_response])
        pred_order_prob = torch.exp(pred_order[torch.arange(pred_order.size(0)), t_order])
        pred_dst_prob = torch.exp(pred_dst[torch.arange(pred_dst.size(0)), t_dst])

        true_response_prob = torch.where(t_response == 1, 1 - t_true_prob[:, 0], t_true_prob[:, 0])
        true_response_prob_loss = F.mse_loss(pred_response_prob, true_response_prob)
        true_order_prob_loss = F.mse_loss(pred_order_prob * pred_dst_prob, t_true_prob[:, 1])

        train_epoch_true_order_prob_loss += true_order_prob_loss.item()
        train_epoch_true_response_prob_loss += true_response_prob_loss.item()

        train_epoch_order_acc += order_acc
        train_epoch_dst_acc += dst_acc
        train_epoch_response_acc += response_acc
        train_epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    results = (train_epoch_loss, train_epoch_order_acc, train_epoch_dst_acc, train_epoch_response_acc,
               train_epoch_true_order_prob_loss, train_epoch_true_response_prob_loss)
    visuals = train_e_chars_list,  train_me_weights_list, train_other_weights_list, train_other_powers_list

    return results, i, visuals, optimizer


def run_experiment(exp_dir, configs):
    visualizer = Visualizer()
    device = configs['device'] if tr.cuda.is_available() else 'cpu'

    if configs['map_type'] == 'standard':
        norm_adjacency = preprocess_adjacency(get_adjacency_matrix('standard'))
        print('adjacency_map', norm_adjacency.shape)
    elif configs['map_type'] == 'small':
        norm_adjacency = preprocess_adjacency(get_adjacency_matrix(BASE_DIR + '/' + configs['map_type']))
        print('adjacency_map', norm_adjacency.shape)
    else:
        norm_adjacency = None

    '''setting env & models'''
    tom_net = FC_PredNet(device=device, norm_adjacency=norm_adjacency, configs=configs)
    tom_net.to(device)

    '''load dataset'''
    if configs['data_dir'] != None:
        dataset_len = configs['last_folder_num']
        eval_dataset = dataset.make_dataset(configs['data_dir'], configs["exp"], configs["input_infos"],
                                            dataset_len - int(dataset_len / 10), dataset_len)

        dataset_len -= int(dataset_len / 10)  # except last test dataset
        dataset_num = [configs['num_data_per_loader']] * (dataset_len // configs['num_data_per_loader'])
        if dataset_len % configs['num_data_per_loader'] != 0:
            dataset_num += [dataset_len % configs['num_data_per_loader']]

    test_loader = DataLoader(eval_dataset, configs['batch_size'], shuffle=False)

    '''set loss and optimizer'''
    criterion = nn.NLLLoss()
    optimizer = tr.optim.Adam(tom_net.parameters(), lr=configs['lr'])

    best_loss = 9999999
    early_stopping = 0
    for epoch in range(configs['num_epoch']):
        '''train'''
        tom_net.train()
        total_train_epoch_order_acc = 0
        total_train_epoch_dst_acc = 0
        total_train_epoch_response_acc = 0
        total_train_epoch_loss = 0
        total_train_epoch_true_order_prob_loss = 0
        total_train_epoch_true_response_prob_loss = 0
        total_train_e_chars_list = []
        total_train_me_weights_list = []
        total_train_other_weights_list = []
        total_train_other_powers_list = []
        total_i = 0
        total_num_data = 0
        start_data_ind = 0
        for num_dataset in dataset_num:
            train_dataset = dataset.make_dataset(configs['data_dir'], configs["exp"], configs["input_infos"],
                                                 start_data_ind, start_data_ind + num_dataset)
            num_data = copy.deepcopy(len(train_dataset))
            total_num_data += num_data
            start_data_ind += num_dataset
            train_loader = DataLoader(train_dataset, configs['batch_size'], shuffle=True)
            results, i, visuals, optimizer = run_one_loader(train_loader, tom_net, criterion, optimizer, device, configs['no_char'])
            train_epoch_loss, train_epoch_order_acc, train_epoch_dst_acc, train_epoch_response_acc, \
                train_epoch_true_order_prob_loss, train_epoch_true_response_prob_loss = results
            train_e_chars_list, train_me_weights_list, train_other_weights_list, train_other_powers_list = visuals
            del train_loader
            del train_dataset

            total_train_epoch_order_acc += train_epoch_order_acc
            total_train_epoch_dst_acc += train_epoch_dst_acc
            total_train_epoch_response_acc += train_epoch_response_acc
            total_train_epoch_loss += train_epoch_loss
            total_train_epoch_true_order_prob_loss += train_epoch_true_order_prob_loss
            total_train_epoch_true_response_prob_loss += train_epoch_true_response_prob_loss

            total_train_e_chars_list += train_e_chars_list
            total_train_me_weights_list += train_me_weights_list
            total_train_other_weights_list += train_other_weights_list
            total_train_other_powers_list += train_other_powers_list
            total_i += i

        total_train_epoch_loss = total_train_epoch_loss / (total_i + 1)
        total_train_epoch_order_acc = total_train_epoch_order_acc / total_num_data
        total_train_epoch_dst_acc = total_train_epoch_dst_acc / total_num_data
        total_train_epoch_response_acc = total_train_epoch_response_acc / total_num_data
        total_train_epoch_true_order_prob_loss = total_train_epoch_true_order_prob_loss / (total_i + 1)
        total_train_epoch_true_response_prob_loss = total_train_epoch_true_response_prob_loss / (total_i + 1)

        if not configs['no_char'] and (epoch % configs['e_char_freq'] == 0):
            concat_e_chars_list = np.concatenate(total_train_e_chars_list, axis=0)
            concat_me_weights_list = np.concatenate(total_train_me_weights_list, axis=0)
            concat_other_weights_list = np.concatenate(total_train_other_weights_list, axis=0)
            train_pil_image_me = visualizer.get_char(concat_e_chars_list, epoch, concat_me_weights_list)
            train_pil_image_other = visualizer.get_char(concat_e_chars_list, epoch, concat_other_weights_list)

            if configs['use_wandb']:
                # wandb.log({'Train Image(me_weight)': wandb.Image(train_pil_image_me)}, step=epoch)
                wandb.log({'Train Image(other_weight)': wandb.Image(train_pil_image_other)}, step=epoch)

        print("Train Epoch: {} | loss: {:3f} | order_acc: {:3f} | dst_acc: {:3f} | response_acc: {:3f}".format(epoch, total_train_epoch_loss,
                                                                                                               total_train_epoch_order_acc,
                                                                                                               total_train_epoch_dst_acc,
                                                                                                               total_train_epoch_response_acc))

        if configs['use_wandb']:
            wandb.log({'Train/loss': total_train_epoch_loss, 'Train/order_acc': total_train_epoch_order_acc,
                       'Train/dst_acc': total_train_epoch_dst_acc,
                       'Train/response_acc': total_train_epoch_response_acc,
                       'Train/true_order_prob_loss': total_train_epoch_true_order_prob_loss,
                       'Train/true_response_prob_loss': total_train_epoch_true_response_prob_loss}, step=epoch)

        '''test'''
        tom_net.eval()
        test_epoch_order_acc = 0
        test_epoch_dst_acc = 0
        test_epoch_response_acc = 0
        test_epoch_loss = 0
        test_epoch_true_order_prob_loss = 0
        test_epoch_true_response_prob_loss = 0
        test_e_chars_list = []
        test_me_weights_list = []
        test_other_weights_list = []

        for i, batch in enumerate(test_loader):
            with tr.no_grad():
                p_board, p_order, p_message, c_board, c_src, c_message, t_order, t_dst, t_response, \
                         me_weights, other_weights, me_ind, other_ind, c_send, c_order, t_true_prob = batch

                p_board = p_board.float().to(device)
                p_order = p_order.float().to(device)
                p_message = p_message.float().to(device)
                c_board = c_board.float().to(device)
                c_src = c_src.float().to(device)
                c_message = c_message.float().to(device)
                c_send = c_send.float().to(device)
                c_order = c_order.float().to(device)
                t_order = tr.flatten(t_order.long().to(device))
                t_dst = tr.flatten(t_dst.long().to(device))
                t_response = tr.flatten(t_response.long().to(device))
                me_ind = me_ind.float().to(device)
                other_ind = other_ind.float().to(device)
                t_true_prob = t_true_prob.to(device)
                pred_order, pred_dst, pred_response, e_char, weights, e_char_raw = tom_net(p_board, p_order, p_message,
                                                                                           c_board, c_src, c_message,
                                                                                           c_send, c_order, other_ind,
                                                                                           me_ind)

                order_loss = criterion(pred_order, t_order)
                dst_loss = criterion(pred_dst, t_dst)
                response_loss = criterion(pred_response, t_response)

                loss = order_loss + dst_loss + response_loss

                if not configs['no_char']:
                    test_e_chars_list.append(e_char.cpu().detach().numpy())
                    test_me_weights_list.append(me_weights.cpu().detach().squeeze(-1).numpy())
                    test_other_weights_list.append(other_weights.cpu().detach().squeeze(-1).numpy())

                pred_order_inds = tr.argmax(pred_order, dim=-1)
                pred_dst_inds = tr.argmax(pred_dst, dim=-1)
                pred_response_inds = tr.argmax(pred_response, dim=-1)

                order_acc = tr.sum(pred_order_inds == t_order).item()
                dst_acc = tr.sum(pred_dst_inds == t_dst).item()
                response_acc = tr.sum(pred_response_inds == t_response).item()

                pred_response_prob = torch.exp(pred_response[torch.arange(pred_response.size(0)), t_response])
                pred_order_prob = torch.exp(pred_order[torch.arange(pred_order.size(0)), t_order])
                pred_dst_prob = torch.exp(pred_dst[torch.arange(pred_dst.size(0)), t_dst])

                true_response_prob = torch.where(t_response == 1, 1 - t_true_prob[:, 0], t_true_prob[:, 0])
                true_response_prob_loss = F.mse_loss(pred_response_prob, true_response_prob)
                true_order_prob_loss = F.mse_loss(pred_order_prob * pred_dst_prob, t_true_prob[:, 1])

                test_epoch_order_acc += order_acc
                test_epoch_dst_acc += dst_acc
                test_epoch_response_acc += response_acc
                test_epoch_loss += loss

                test_epoch_true_order_prob_loss += true_order_prob_loss.item()
                test_epoch_true_response_prob_loss += true_response_prob_loss.item()

        test_epoch_order_acc /= len(eval_dataset)
        test_epoch_dst_acc /= len(eval_dataset)
        test_epoch_response_acc /= len(eval_dataset)
        test_epoch_loss = test_epoch_loss / (i + 1)
        test_epoch_true_order_prob_loss = test_epoch_true_order_prob_loss / (i + 1)
        test_epoch_true_response_prob_loss = test_epoch_true_response_prob_loss / (i + 1)

        if not configs['no_char'] and (epoch % configs['e_char_freq'] == 0):
            concat_test_e_chars_list = np.concatenate(test_e_chars_list, axis=0)
            concat_test_me_weights_list = np.concatenate(test_me_weights_list, axis=0)
            concat_test_other_weights_list = np.concatenate(test_other_weights_list, axis=0)
            test_pil_image_me = visualizer.get_char(concat_test_e_chars_list, epoch, concat_test_me_weights_list)
            test_pil_image_other = visualizer.get_char(concat_test_e_chars_list, epoch, concat_test_other_weights_list)

            if configs['use_wandb']:
                # wandb.log({'Test Image(me_weight)': wandb.Image(test_pil_image_me)}, step=epoch)
                wandb.log({'Test Image(other_weight)': wandb.Image(test_pil_image_other)}, step=epoch)

        print("Test Epoch: {} | loss: {:3f} | order_acc: {:3f} | dst_acc: {:3f} | response_acc: {:3f}".format(epoch, test_epoch_loss, test_epoch_order_acc, test_epoch_dst_acc, test_epoch_response_acc))
        print("Test Epoch: {} | true_order_loss: {:3f} | true_response_loss: {:3f} ".format(epoch,
                                                                                            test_epoch_true_order_prob_loss,
                                                                                            test_epoch_true_response_prob_loss))

        if configs['use_wandb']:
            wandb.log({'Test/loss': test_epoch_loss, 'Test/order_acc': test_epoch_order_acc,
                       'Test/dst_acc': test_epoch_dst_acc, 'Test/response_acc': test_epoch_response_acc,
                       'Test/true_order_prob_loss': test_epoch_true_order_prob_loss,
                       'Test/true_response_prob_loss': test_epoch_true_response_prob_loss}, step=epoch)

        if epoch % configs['save_freq'] == 0 and configs['save_model'] and epoch >= configs['save_start_epoch']:
            if test_epoch_loss < best_loss:
                print('Best performance at epoch: {}'.format(epoch))
                print('Save model in', exp_dir)
                best_loss = test_epoch_loss
                model_info = dict(epoch=epoch, exp=configs["exp"], seed=configs['seed'])
                save_model(tom_net, exp_dir, model_info)
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping >= configs["early_stop_num"]:
                    if configs['use_wandb']:
                        wandb.finish()
                    break
