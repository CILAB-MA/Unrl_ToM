import wandb
import copy
import os
from torch.utils.data import DataLoader

from experiments import dataset
from experiments.tools import *
from utils.visualize import Visualizer
from .PredNet import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

no_message_pred = False

def run_one_loader(train_loader, tom_net, criterion, optimizer, no_char=False):
    train_epoch_order_acc = 0
    train_epoch_dst_acc = 0
    train_epoch_response_acc = 0
    train_epoch_loss = 0
    train_e_chars_list = []
    train_me_weights_list = []
    train_other_weights_list = []

    for i, batch in enumerate(train_loader):
        c_board, c_src, c_message, t_order, t_dst, t_response, \
        me_weights, other_weights, me_ind, other_ind, c_send, c_order, c_other_internal = batch

        c_board = c_board.float().cuda()
        c_src = c_src.float().cuda()
        c_message = c_message.float().cuda()
        c_send = c_send.float().cuda()
        c_order = c_order.float().cuda()
        c_other_internal = c_other_internal.float().cuda()
        t_order = tr.flatten(t_order.long().cuda())
        t_dst = tr.flatten(t_dst.long().cuda())
        t_response = tr.flatten(t_response.long().cuda())
        pred_order, pred_dst, pred_response, e_char = tom_net(c_board, c_src, c_message, c_send,  c_order, me_weights,
                                                              other_weights, me_ind, other_ind, c_other_internal)
        order_loss = criterion(pred_order, t_order)
        dst_loss = criterion(pred_dst, t_dst)
        if no_message_pred:
            loss = order_loss + dst_loss
        else:
            response_loss = criterion(pred_response, t_response)
            loss = order_loss + dst_loss + response_loss

        train_e_chars_list.append(e_char.cpu().detach().numpy())
        train_me_weights_list.append(me_weights.cpu().detach().squeeze(-1).numpy())
        train_other_weights_list.append(other_weights.cpu().detach().squeeze(-1).numpy())

        pred_order_inds = tr.argmax(pred_order, dim=-1)
        pred_dst_inds = tr.argmax(pred_dst, dim=-1)
        if not no_message_pred:
            pred_response_inds = tr.argmax(pred_response, dim=-1)

        order_acc = tr.sum(pred_order_inds == t_order).item()
        dst_acc = tr.sum(pred_dst_inds == t_dst).item()
        if not no_message_pred:
            response_acc = tr.sum(pred_response_inds == t_response).item()

        train_epoch_order_acc += order_acc
        train_epoch_dst_acc += dst_acc
        if not no_message_pred:
            train_epoch_response_acc += response_acc
        train_epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    results = train_epoch_loss, train_epoch_order_acc, train_epoch_dst_acc, train_epoch_response_acc
    visuals = train_e_chars_list,  train_me_weights_list, train_other_weights_list

    return results, i, visuals, optimizer

def run_experiment(exp_dir, configs):
    visualizer = Visualizer()
    device = 'cuda' if tr.cuda.is_available() else 'cpu'

    if configs['map_type'] == 'standard':
        norm_adjacency = get_adjacency_matrix('standard')
        print('adjacency_map', norm_adjacency.shape)
    elif configs['map_type'] == 'small':
        norm_adjacency = get_adjacency_matrix(BASE_DIR + '/' + configs['map_type'])
        print('adjacency_map', norm_adjacency.shape)

    '''setting env & models'''
    tom_net = FC_PredNet(device=device, norm_adjacency=norm_adjacency, configs=configs)
    # if configs['prednet_type'] == 'fc':
    #     tom_net = FC_PredNet(device=device, norm_adjacency=norm_adjacency, configs=configs)
    # elif configs['prednet_type'] == 'only_lstm':
    #     tom_net = Only_lstm_PredNet(device=device, norm_adjacency=norm_adjacency, configs=configs)
    # else:
    #     ValueError("SET PREDICTION NETWORK FOR ORACLE.")
    tom_net.to(device)

    '''load dataset'''
    if configs['data_dir'] != None:
        dataset_len = configs['last_folder_num']
        eval_dataset = dataset.make_dataset(configs['data_dir'], configs["exp"], configs["input_infos"],
                                                dataset_len - 2, dataset_len)

        dataset_len -= 2 # except last test dataset
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
        total_train_e_chars_list = []
        total_train_me_weights_list = []
        total_train_other_weights_list = []
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
            results, i, visuals, optimizer = run_one_loader(train_loader, tom_net, criterion, optimizer, configs['no_char'])
            train_epoch_loss, train_epoch_order_acc, train_epoch_dst_acc, train_epoch_response_acc = results
            train_e_chars_list, train_me_weights_list, train_other_weights_list = visuals
            del train_loader
            del train_dataset

            total_train_epoch_order_acc += train_epoch_order_acc
            total_train_epoch_dst_acc += train_epoch_dst_acc
            total_train_epoch_response_acc += train_epoch_response_acc
            total_train_epoch_loss += train_epoch_loss

            total_train_e_chars_list += train_e_chars_list
            total_train_me_weights_list += train_me_weights_list
            total_train_other_weights_list += train_other_weights_list
            total_i += i

        total_train_epoch_loss = total_train_epoch_loss / (total_i + 1)
        total_train_epoch_order_acc = total_train_epoch_order_acc / total_num_data
        total_train_epoch_dst_acc = total_train_epoch_dst_acc / total_num_data
        total_train_epoch_response_acc = total_train_epoch_response_acc / total_num_data

        if epoch % configs['e_char_freq'] == 0:
            e_chars_list = np.concatenate(total_train_e_chars_list, axis=0)
            me_weights_list = np.concatenate(total_train_me_weights_list, axis=0)
            other_weights_list = np.concatenate(total_train_other_weights_list, axis=0)
            pil_image_me = visualizer.get_char(e_chars_list, epoch, me_weights_list)
            pil_image_other = visualizer.get_char(e_chars_list, epoch, other_weights_list)
            if configs['use_wandb']:
                wandb.log({'Train Image(me_weight)': wandb.Image(pil_image_me)}, step=epoch)
                wandb.log({'Train Image(other_weight)': wandb.Image(pil_image_other)}, step=epoch)

        print("Train Epoch: {} | loss: {:3f} | order_acc: {:3f} | dst_acc: {:3f} | response_acc: {:3f}".format(epoch, total_train_epoch_loss,
                                                                                                               total_train_epoch_order_acc,
                                                                                                               total_train_epoch_dst_acc,
                                                                                                               total_train_epoch_response_acc))

        if configs['use_wandb']:
            wandb.log({'Train_loss': total_train_epoch_loss, 'Train_order_acc': total_train_epoch_order_acc,
                       'Train_dst_acc': total_train_epoch_dst_acc, 'Train_response_acc': total_train_epoch_response_acc},
                      step=epoch)

        '''test'''
        tom_net.eval()
        test_epoch_order_acc = 0
        test_epoch_dst_acc = 0
        test_epoch_response_acc = 0
        test_epoch_loss = 0
        test_e_chars_list = []
        test_me_weights_list = []
        test_other_weights_list = []

        for i, batch in enumerate(test_loader):
            with tr.no_grad():
                c_board, c_src, c_message, t_order, t_dst, t_response, \
                me_weights, other_weights, me_ind, other_ind, c_send, c_order, c_other_internal = batch

                c_board = c_board.float().cuda()
                c_src = c_src.float().cuda()
                c_message = c_message.float().cuda()
                c_send = c_send.float().cuda()
                c_order = c_order.float().cuda()
                c_other_internal = c_other_internal.float().cuda()
                t_order = tr.flatten(t_order.long().cuda())
                t_dst = tr.flatten(t_dst.long().cuda())
                t_response = tr.flatten(t_response.long().cuda())
                pred_order, pred_dst, pred_response, e_char = tom_net(c_board, c_src, c_message, c_send,  c_order,
                                                                      me_weights, other_weights, me_ind, other_ind,
                                                                      c_other_internal)

                order_loss = criterion(pred_order, t_order)
                dst_loss = criterion(pred_dst, t_dst)
                if no_message_pred:
                    loss = order_loss + dst_loss
                else:
                    response_loss = criterion(pred_response, t_response)
                    loss = order_loss + dst_loss + response_loss

                test_e_chars_list.append(e_char.cpu().detach().numpy())
                test_me_weights_list.append(me_weights.cpu().detach().squeeze(-1).numpy())
                test_other_weights_list.append(other_weights.cpu().detach().squeeze(-1).numpy())

                pred_order_inds = tr.argmax(pred_order, dim=-1)
                pred_dst_inds = tr.argmax(pred_dst, dim=-1)
                if not no_message_pred:
                    pred_response_inds = tr.argmax(pred_response, dim=-1)

                order_acc = tr.sum(pred_order_inds == t_order).item()
                dst_acc = tr.sum(pred_dst_inds == t_dst).item()
                if not no_message_pred:
                    response_acc = tr.sum(pred_response_inds == t_response).item()

                test_epoch_order_acc += order_acc
                test_epoch_dst_acc += dst_acc
                if not no_message_pred:
                    test_epoch_response_acc += response_acc
                test_epoch_loss += loss

        test_epoch_order_acc /= len(eval_dataset)
        test_epoch_dst_acc /= len(eval_dataset)
        test_epoch_response_acc /= len(eval_dataset)
        test_epoch_loss = test_epoch_loss / (i + 1)

        if epoch % configs['e_char_freq'] == 0:
            test_e_chars_list = np.concatenate(test_e_chars_list, axis=0)
            test_me_weights_list = np.concatenate(test_me_weights_list, axis=0)
            test_other_weights_list = np.concatenate(test_other_weights_list, axis=0)
            pil_image_me = visualizer.get_char(test_e_chars_list, epoch, test_me_weights_list)
            pil_image_other = visualizer.get_char(test_e_chars_list, epoch, test_other_weights_list)
            if configs['use_wandb']:
                wandb.log({'Test Image(me_weight)': wandb.Image(pil_image_me)}, step=epoch)
                wandb.log({'Test Image(other_weight)': wandb.Image(pil_image_other)}, step=epoch)

        print("Test Epoch: {} | loss: {:3f} | order_acc: {:3f} | dst_acc: {:3f} | response_acc: {:3f}".format(epoch, test_epoch_loss, test_epoch_order_acc, test_epoch_dst_acc, test_epoch_response_acc))
        if configs['use_wandb']:
            wandb.log({'Test_loss': test_epoch_loss, 'Test_order_acc': test_epoch_order_acc, 'Test_dst_acc': test_epoch_dst_acc, 'Test_response_acc': test_epoch_response_acc}, step=epoch)

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
                    print("Stop the training early at epoch {}".format(epoch))
                    break