#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn

# use wandb to track loss and accuracy
import wandb

# loss used in our work
def entropy(output):
    probs = torch.softmax(output, 1)
    log_probs = torch.log_softmax(output, 1)
    entropies = -torch.sum(probs * log_probs, 1)
    return torch.mean(entropies)


def reconsturction_loss(output, target):
    return nn.MSELoss()(output, target)


def action_classification_loss(output, target):
    action_classification_loss = nn.CrossEntropyLoss()(output, target)
    return action_classification_loss


def privacy_classification_loss(output, target):
    privacy_loss = nn.CrossEntropyLoss()(output, target)
    return privacy_loss


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Skeleton Anonymization Framework with Shift-GCN')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='./config/train_adver_resnet.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder_anonymization', help='data loader will be used')
    parser.add_argument(
        '--test-feeder', default='feeder.feeder_anonymization', help='test data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=140,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)


    
    parser.add_argument('--wandb', type=str, default="Skeleton_anonymization", help='project name of wandb')
    parser.add_argument('--entity', type=str, default="user", help='entity of wandb')

    parser.add_argument(
        '--anonymizer-model',
        default=None,
        help='the anonymizer model will be used')
    parser.add_argument(
        '--action-model',
        default=None,
        help='the action model will be used')
    parser.add_argument(
        '--privacy-model',
        default=None,
        help='the privacy model will be used')

    parser.add_argument(
        '--action-model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--privacy-model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')

    parser.add_argument(
        '--test-action',
        type=dict,
        default=dict())
    parser.add_argument(
        '--test-privacy',
        type=dict,
        default=dict())

    parser.add_argument(
        '--minimization-steps',
        type=int,
        default=[0, 1, 2],
        nargs='+',
        help='steps to train anonymizer')
    parser.add_argument(
        '--alpha',
        type=float,)
    parser.add_argument(
        '--beta',
        type=float,)

    parser.add_argument(
        '--pretrained-action',
        type=str,)
    parser.add_argument(
        '--pretrained-privacy',
        type=str,)
    parser.add_argument(
        '--pretrained-privacy-test',
        nargs='+',
        type=str,)
    parser.add_argument(
        '--pretrained-anonymizer',
        default=None)
    parser.add_argument(
        '--dataset',
        type=str,
        default='ntu60')

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):

        arg.model_saved_name = "./save_models/"+arg.Experiment_name
        arg.work_dir = "./work_dir/"+arg.Experiment_name
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)

        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.test_loader_action = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_action['batch_size'],
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        self.test_loader_privacy = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_privacy['batch_size'],
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed) 

    def load_eval_action_model(self, weight):
        self.print_log("Using action weight %s" % weight)
        self.eval_action_model = import_class(self.arg.action_model)(
            **self.arg.action_model_args).cuda(self.output_device)
        self.eval_action_model.load_state_dict(torch.load(weight))

        self.eval_action_model = nn.DataParallel(
            self.eval_action_model,
            device_ids=self.arg.device,
            output_device=self.output_device
        )
        self.eval_action_model.eval()

    def load_eval_privacy_model(self, weight):
        self.print_log("Using privacy weight %s" % weight)
        self.eval_privacy_model = import_class(self.arg.privacy_model)(
            **self.arg.privacy_model_args).cuda(self.output_device)
        self.eval_privacy_model.load_state_dict(torch.load(weight))
        self.eval_privacy_model = nn.DataParallel(
            self.eval_privacy_model,
            device_ids=self.arg.device,
            output_device=self.output_device
        )
        self.eval_privacy_model.eval()

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device 

        AnonymizerModel = import_class(self.arg.anonymizer_model)
        shutil.copy2(inspect.getfile(AnonymizerModel), self.arg.work_dir)

        self.anonymizer = AnonymizerModel(**self.arg.model_args).cuda(output_device)
        self.action_classifier = import_class(self.arg.action_model)(**self.arg.action_model_args).cuda(self.output_device)
        self.privacy_classifier = import_class(self.arg.privacy_model)(**self.arg.privacy_model_args).cuda(self.output_device)

        if self.arg.pretrained_action:
            self.print_log("Using pretrained action model %s" %
                           self.arg.pretrained_action)
            self.action_classifier.load_state_dict(
                torch.load(self.arg.pretrained_action))

        if self.arg.pretrained_privacy:
            self.print_log("Using pretrained privacy model %s" %
                           self.arg.pretrained_privacy)
            self.privacy_classifier.load_state_dict(
                torch.load(self.arg.pretrained_privacy))

        self.action_classifier.eval()

        self.print_log("Loading models for evaluation")
        self.load_eval_action_model(self.arg.pretrained_action)
        self.load_eval_privacy_model(self.arg.pretrained_privacy_test)

        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.anonymizer.load_state_dict(weights)
            except:
                state = self.anonymizer.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.anonymizer.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.anonymizer = nn.DataParallel(
                    self.anonymizer,
                    device_ids=self.arg.device,
                    output_device=output_device)
                self.action_classifier = nn.DataParallel(
                    self.action_classifier,
                    device_ids=self.arg.device,
                    output_device=output_device)
                self.privacy_classifier = nn.DataParallel(
                    self.privacy_classifier,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':

            #Anonymizer
            params_dict_anon = dict(self.anonymizer.named_parameters())
            params_anon = []

            for key, value in params_dict_anon.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4
                if 'Linear_weight' in key:
                    weight_decay = 1e-3
                elif 'Mask' in key:
                    weight_decay = 0.0
                    
                params_anon += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult, 'weight_decay': weight_decay}]
            self.anonymizer_optimizer = optim.SGD(
                params_anon,
                momentum=0.9,
                nesterov=self.arg.nesterov)


            #Privacy classifier
            params_dict_privacy = dict(self.privacy_classifier.named_parameters())
            params_privacy = []

            for key, value in params_dict_privacy.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4
                if 'Linear_weight' in key:
                    weight_decay = 1e-3
                elif 'Mask' in key:
                    weight_decay = 0.0
                    
                params_privacy += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult, 'weight_decay': weight_decay}]
            self.privacy_classifier_optimizer = optim.SGD(
                params_privacy,
                momentum=0.9,
                nesterov=self.arg.nesterov)
            
        elif self.arg.optimizer == 'Adam':
            self.anonymizer_optimizer = optim.Adam(
                self.anonymizer.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

            self.privacy_classifier_optimizer = optim.Adam(
                self.privacy_classifier.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

        else:
            raise ValueError()

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            os.makedirs(self.arg.work_dir+'/eval_results')
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.anonymizer_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in self.privacy_classifier_optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    # minimization step
    def anonymizer_step(self, data, epoch, privacy_label, action_label,
                        recon_loss_list, action_loss_list, privacy_loss_list,
                        action_acc_list, privacy_acc_list, total_loss_list, timer):
        self.anonymizer_optimizer.zero_grad()
        anonymized = self.anonymizer(data)
        action = self.action_classifier(anonymized)

        self.network_time = time.time()-self.start

        action_loss = action_classification_loss(action, action_label)
        _, predict_action = torch.max(action, 1)
        action_acc = torch.mean((predict_action == action_label).float())

        privacy = self.privacy_classifier(anonymized)

        privacy_loss = entropy(privacy)
        _, predict_privacy = torch.max(privacy, 1)
        privacy_acc = torch.mean((predict_privacy ==
                                  privacy_label).float()).item()

        recon_loss = reconsturction_loss(anonymized, data)

        anonymization_loss = action_loss - \
            self.arg.alpha * privacy_loss + self.arg.beta * recon_loss

        anonymization_loss.backward()
        self.anonymizer_optimizer.step()

        recon_loss_list.append(recon_loss.item())
        total_loss_list.append(anonymization_loss.item())
        action_loss_list.append(action_loss.item())
        privacy_loss_list.append(privacy_loss.item())

        action_acc_list.append(action_acc.item())
        privacy_acc_list.append(privacy_acc)
        


    # maximization step
    def privacy_classifier_step(self, data, epoch, privacy_label, privacy_loss_list, privacy_acc_list, timer):
        self.privacy_classifier_optimizer.zero_grad()

        anonymized = self.anonymizer(data)
        privacy = self.privacy_classifier(anonymized)

        self.network_time = time.time()-self.start

        privacy_loss = privacy_classification_loss(privacy, privacy_label)
        _, predict_privacy = torch.max(privacy, 1)
        privacy_acc = torch.mean(
            (predict_privacy == privacy_label).float()).item()
        privacy_loss.backward()
        self.privacy_classifier_optimizer.step()

        privacy_loss_list.append(privacy_loss.item())
        privacy_acc_list.append(privacy_acc)
       
        

    def train(self, epoch, save_model=False):
        self.anonymizer.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, dynamic_ncols=True)

        total_loss_list = []
        recon_loss_list = []
        action_loss_list = []
        privacy_loss_list = []
        privacy_acc_list = []
        action_acc_list = []

        if epoch >= self.arg.only_train_epoch:
            for key, value in self.anonymizer.named_parameters():
                if 'PA' in key:
                    value.requires_grad = True
                    print(key + '-require grad')
        else:
            for key, value in self.anonymizer.named_parameters():
                if 'PA' in key:
                    value.requires_grad = False
                    print(key + '-not require grad')


        for batch_idx, (data, privacy_label, action_label, index) in enumerate(process):
            self.global_step += 1
            # get data
            data = data.float().cuda(self.output_device)
            privacy_label = privacy_label.long().cuda(self.output_device)
            action_label = action_label.long().cuda(self.output_device)

            timer['dataloader'] += self.split_time()

            # forward
            self.start = time.time()
            
            if (self.global_step % (self.arg.minimization_steps + 1) == 0): #maximization(privacy) step
                self.privacy_classifier_step(data, epoch, privacy_label, privacy_loss_list, privacy_acc_list, timer)
            else: #minimization(anonimization) step
                self.anonymizer_step(data, epoch, privacy_label, action_label, recon_loss_list,
                                     action_loss_list, privacy_loss_list, action_acc_list, privacy_acc_list, total_loss_list, timer)

            timer['model'] += self.split_time()

            mean_recon_loss = np.mean(recon_loss_list) if len(
                recon_loss_list) else np.nan
            mean_action_loss = np.mean(action_loss_list) if len(
                action_loss_list) else np.nan
            mean_action_acc = np.mean(action_acc_list) if len(
                action_acc_list) else np.nan
            mean_privacy_loss = np.mean(privacy_loss_list) if len(
                privacy_loss_list) else np.nan
            mean_privacy_acc = np.mean(privacy_acc_list) if len(
                privacy_acc_list) else np.nan

            

            self.lr = self.anonymizer_optimizer.param_groups[0]['lr']

            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. recon_loss: {:.4f}, action_loss: {:.4f}, priv_loss: {:.4f}, action_acc:{:.4f}, priv_acc: {:.4f}  lr:{:.6f}  network_time: {:.4f}'.format(
                        batch_idx, len(loader), mean_recon_loss, mean_action_loss,
                        mean_privacy_loss, mean_action_acc, mean_privacy_acc, self.lr, self.network_time))

            timer['statistics'] += self.split_time()

        wandb.log({"action_acc": np.mean(action_acc_list)}, step=epoch) 
        wandb.log({"recon_loss": np.mean(recon_loss_list)}, step=epoch) 
        wandb.log({"action_loss": np.mean(action_loss_list)}, step=epoch) 
        wandb.log({"privacy_loss": np.mean(privacy_loss_list)}, step=epoch) 
        wandb.log({"privacy_acc": np.mean(privacy_acc_list)}, step=epoch)
        wandb.log({"total_loss": np.mean(total_loss_list)}, step=epoch)

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }


        if save_model:
            state_dict = self.anonymizer.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')


    def eval_action_validate(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.eval_action_model.eval()
        self.print_log(f'Action: eval test')
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_values = []
            action_batches = []
            labels = []

            step = 0
            process = tqdm(self.test_loader_action, dynamic_ncols=True)
            with torch.no_grad():
                for batch_idx, (data, privacy_label, action_label, index) in enumerate(process):
                    
                    labels.extend(action_label.cpu().tolist())
                    action_label = action_label.long().cuda(self.output_device)

                    
                    anonymized = self.anonymizer(data)
                    #anonymized = np.load('/data_seoul/saemi/BASAR-Black-box-Attack-on-Skeletal-Action-Recognition/results/ntu/untargeted/0705_184001/target_samples.npy') 
                    #anonymized = data + (0.001**0.5)*torch.randn(664, 25, 1)
                    #anonymized = torch.Tensor(anonymized)
                    #anonymized = data

                    action = self.eval_action_model(anonymized)

                    loss = action_classification_loss(action, action_label)
                    loss_values.append(loss.item())
                    action_batches.append(action.data.cpu().numpy())

                    step += 1

            score = np.concatenate(action_batches)
            loss = np.mean(loss_values)

            accuracy = self.test_loader_action.dataset.top_k_action(score, 1)
            self.accuracy_total.append(accuracy)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                score_dict = dict(
                    zip(self.test_loader_action.dataset.sample_name, score))

                with open('./work_dir/' + arg.Experiment_name + '/eval_results/best_acc' +'.pkl'.format(
                        epoch, accuracy), 'wb') as f:
                    pickle.dump(score_dict, f)

            print('Eval Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            wandb.log({
                    "val_action_acc_top1": accuracy,
                    "val_action_acc_top5": 100 * self.test_loader_action.dataset.top_k_action(score, 5),
                    "val_action_loss": np.mean(loss_values),
                    "val_recon_loss": np.square(anonymized.cpu().numpy() - data.cpu().numpy()).mean(), 
                    }, step=epoch) 

            score_dict = dict(
                zip(self.test_loader_action.dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.test_loader_action), np.mean(loss_values)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.test_loader_action.dataset.top_k_action(score, k)))

            with open('./work_dir/' + arg.Experiment_name + '/eval_results/epoch_' + str(epoch) + '_' + str(accuracy) + '.pkl'.format(
                    epoch, accuracy), 'wb') as f:
                pickle.dump(score_dict, f)

    def eval_privacy_validate(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.eval_privacy_model.eval()
        self.print_log(f'Privacy: eval test')
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_values = []
            privacy_batches = []
            labels = []

            step = 0
            process = tqdm(self.test_loader_privacy, dynamic_ncols=True)
            with torch.no_grad():
                for batch_idx, (data, privacy_label, action_label, index) in enumerate(process):
                    labels.extend(privacy_label.cpu().tolist())
                    privacy_label = privacy_label.long().cuda(self.output_device)
                
                    anonymized = self.anonymizer(data)
                    #anonymized = data + (0.001**0.5)*torch.randn(664, 25, 1)
                    #anonymized = np.load('/data_seoul/saemi/BASAR-Black-box-Attack-on-Skeletal-Action-Recognition/results/ntu/untargeted/0705_184001/target_samples.npy') 
                    #anonymized = torch.Tensor(anonymized)
                    #anonymized = data
                    privacy = self.eval_privacy_model(anonymized)
                        
                    loss = entropy(privacy)
                    loss_values.append(loss.item())
                    privacy_batches.append(privacy.data.cpu().numpy())

                    step += 1
                
        
            score = np.concatenate(privacy_batches)

            accuracy = self.test_loader_privacy.dataset.top_k_privacy(score, 1)
            self.accuracy_total.append(accuracy)

            if accuracy > self.best_acc:
                self.best_acc = accuracy
                score_dict = dict(
                    zip(self.test_loader_privacy.dataset.sample_name, score))

                with open('./work_dir/' + arg.Experiment_name + '/eval_results/best_acc' +'.pkl'.format(
                        epoch, accuracy), 'wb') as f:
                    pickle.dump(score_dict, f)

            print('Eval Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            wandb.log({
                    "val_privacy_top1": accuracy,
                    "val_privacy_top5": 100 * self.test_loader_privacy.dataset.top_k_privacy(score, 5),
                    "val_privacy_loss": np.mean(loss_values),
                    }, step=epoch)

            score_dict = dict(
                zip(self.test_loader_privacy.dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.test_loader_privacy), np.mean(loss_values)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.test_loader_privacy.dataset.top_k_privacy(score, k)))

            with open('./work_dir/' + arg.Experiment_name + '/eval_results/epoch_' + str(epoch) + '_' + str(accuracy) + '.pkl'.format(
                    epoch, accuracy), 'wb') as f:
                pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            wandb.watch(self.anonymizer, log_freq=10)
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                self.train(epoch, save_model=True)

                self.accuracy_total = []
                self.eval_action_validate(epoch=epoch, save_score=self.arg.save_score, loader_name=['test'])
                self.eval_privacy_validate(epoch=epoch, save_score=self.arg.save_score, loader_name=['test'])

                wandb.log({
                    "val_area": self.accuracy_total[0] * (1-self.accuracy_total[1]), 
                    }, step=epoch) 

            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.anonymizer_model))
            self.print_log('Weights: {}.'.format(self.arg.weights))

            self.accuracy_total = []
            self.eval_action_validate(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.eval_privacy_validate(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0) 

    #wandb initialization
    wandb.init(project=arg.wandb, entity=arg.entity)
    wandb.config.update(arg) 


    wandb.define_metric("val_action_acc_top1", summary="max")
    wandb.define_metric("val_action_acc_top5", summary="max")
    wandb.define_metric("val_action_loss", summary="min")
    wandb.define_metric("val_recon_loss", summary="min")
    wandb.define_metric("val_privacy_top1", summary="min")
    wandb.define_metric("val_privacy_top5", summary="min")

    processor = Processor(arg)
    processor.start()
