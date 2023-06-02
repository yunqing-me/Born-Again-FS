import numpy as np
import os
import glob
import torch
import argparse


def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % script)
    parser.add_argument('--dataset', default='multi',
                        help='miniImagenet/cub/cars/places/plantae, specify multi for training with multiple domains')
    parser.add_argument('--testset', default='places')
    parser.add_argument('--model', default='ResNet10', help='model: Conv{4|6} / ResNet{10|18|34}')
    parser.add_argument('--method', default='matchingnet')
    parser.add_argument('--train_n_way', default=5, type=int, help='class num to classify for training')
    parser.add_argument('--test_n_way', default=5, type=int, help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--train_aug', action='store_true', help='perform data augmentation or not during training ')
    parser.add_argument('--name', default='tmp', type=str, help='')
    parser.add_argument('--teacher', default='ResNet10', type=str, help='')
    parser.add_argument('--save_dir', default='../../output', type=str, help='')
    parser.add_argument('--data_dir', default='../../datasets', type=str, help='')

    if script == 'train':
        parser.add_argument('--save_freq', default=25, type=int, help='Save frequency')
        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=800, type=int, help='Stopping epoch')
        parser.add_argument('--warmup', default='baseline++')
    else:
        raise ValueError('Unknown script')
    return parser.parse_args()


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir, resume_epoch=-1):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None
    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    epoch = max_epoch if resume_epoch == -1 else resume_epoch
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def load_warmup_state(filename, method):
    print('load pre-trained model file: {}'.format(filename))
    warmup_resume_file = get_resume_file(filename)
    tmp = torch.load(warmup_resume_file)
    if tmp is not None:
        state = tmp['state']
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if method == 'matchingnet' and 'feature.' in key and '.7.' not in key:
                newkey = key.replace("feature.", "")
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
    else:
        raise ValueError('No pre-trained encoder file found!')
    return state


def simple_load_model(params, dataset, teacher_model):
    """
    directly load model with state
    """
    tmp_method = 'matchingnet'
    print(f'\nLoad teacher model of {dataset}...')
    if 'matchingnet' in params.method:
        tmp_method = 'matchingnet'

    name_teacher_model = f'single_{tmp_method}_{dataset}_shot_{params.n_shot}'
    teacher_modelfile = os.path.join('%s/checkpoints/%s' % (params.save_dir, name_teacher_model), 'best_model.tar')
    teacher_state = torch.load(teacher_modelfile)['state']
    teacher_model.load_state_dict(teacher_state, strict=False)
    print(f'successfully load teacher model of {dataset} with {tmp_method}!')
    return teacher_model


def simple_load_teacher(params, dataset, teacher_model):
    """
    directly load model with state
    """
    tmp_method = 'matchingnet'
    if params.teacher == 'Conv4':
        model = 'conv4'
    elif params.teacher == 'ResNet18':
        model = 'resnet18'
    elif params.teacher == 'ResNet10':
        model = 'resnet10'
    print(f'\nLoad teacher model of {dataset}...')
    if 'matchingnet' in params.method:
        tmp_method = 'matchingnet'

    if model != 'resnet10':
        name_teacher_model = f'single_{tmp_method}_{dataset}_shot_{params.n_shot}_{model}'
    else: 
        name_teacher_model = f'single_{tmp_method}_{dataset}_shot_{params.n_shot}'
        
    teacher_modelfile = os.path.join('%s/checkpoints/%s' % (params.save_dir, name_teacher_model), 'best_model.tar')
    teacher_state = torch.load(teacher_modelfile)['state']
    teacher_model.load_state_dict(teacher_state, strict=False)
    print(f'successfully load teacher model of {dataset} with {tmp_method}!')
    return teacher_model