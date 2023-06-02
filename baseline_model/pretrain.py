import torch
import torch.optim
import numpy as np
import os
from methods.backbone import model_dict
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from options import parse_args
import torchmeta


def train(base_loader, model, start_epoch, stop_epoch, params):
    optimizer = torch.optim.Adam(model.parameters())
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    total_it = 0

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        total_it = model.train_loop(epoch, base_loader, optimizer, total_it)

        # no validation at pre-train stage
        if ((epoch + 1) % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch+1))
            torch.save({'epoch': epoch+1, 'state': model.state_dict()}, outfile)
    return model


# --- main function ---
if __name__ == '__main__':
    model = None
    np.random.seed(1)

    # parser argument
    params = parse_args('train')
    print('--- baseline training: {} ---\n'.format(params.name))
    print(params)

    # output and tensorboard dir
    params.tf_dir = '%s/log/%s' % (params.save_dir, params.name)
    params.checkpoint_dir = '%s/checkpoints/%s' % (params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- prepare dataloader ---')
    params.dataset = 'miniImagenet'  # pretrain the model on miniImagenet
    print('pretrain on miniImagenet')
    base_file = os.path.join(params.data_dir, params.dataset, 'base.json')

    # model
    print('\n--- build model ---')
    if 'Conv' in params.model:
        image_size = 84
    else:
        image_size = 224

    if params.method in ['baseline', 'baseline++']:
        print('pre-training the feature encoder {} using method {}'.format(params.model, params.method))
        base_datamgr = SimpleDataManager(image_size, batch_size=16)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
        if params.method == 'baseline':
            model = BaselineTrain(model_dict[params.model], params.num_classes, tf_path=params.tf_dir)
        elif params.method == 'baseline++':
            model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist', tf_path=params.tf_dir)
    else:
        raise ValueError('This is only for pre-train')

    model = model.cuda()
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    print('\n--- start pretraining ...---')
    model = train(base_loader, model, start_epoch, stop_epoch, params)

