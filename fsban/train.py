import numpy as np
import os
import random
import torch
from data.datamgr import SetDataManager, SimpleDataManager
from options import parse_args, get_resume_file, load_warmup_state, simple_load_model, simple_load_teacher
from methods.meta_distill import MetaDistill
from methods.matchingnet import MatchingNet
from methods.backbone import model_dict
from methods import backbone
# import torchmeta


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


# training iterations
def train(base_datamgr, datasets, val_loader, model, start_epoch, stop_epoch, params):
    # for validation
    max_acc = 0
    total_it = 0

    # training
    for epoch in range(start_epoch, stop_epoch):

        # randomly split seen domains to pseudo-seen and pseudo-unseen domains
        if epoch < 10:
            random_set = ['miniImagenet', 'miniImagenet']
        else:
            random_set = random.sample(datasets, k=2)
        # random_set = random.sample(datasets, k=2)
        ps_set = random_set[0]
        pu_set = random_set[1:]  # consider 1 pu set in tmp
        ps_loader = base_datamgr.get_data_loader(os.path.join(params.data_dir, ps_set, 'base.json'),
                                                 aug=params.train_aug)
        pu_loader = base_datamgr.get_data_loader(
            [os.path.join(params.data_dir, dataset, 'base.json') for dataset in pu_set], aug=params.train_aug)

        # train loop
        model.train()
        total_it = model.trainall_loop(epoch, ps_set, pu_set[0], ps_loader, pu_loader, total_it)

        # validation using 4 seen domains
        model.eval()
        with torch.no_grad():
            acc = model.test_loop(val_loader)
        # save model
        if acc > max_acc:
            print("best model found, saving...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch+1, 'state': model.state_dict()}, outfile)
        else:
            print("best accuracy {:f}".format(max_acc))

        if ((epoch + 1) % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch+1))
            torch.save({'epoch': epoch+1, 'state': model.state_dict()}, outfile)
    return


# --- main function ---
if __name__ == '__main__':
    model = None  # init student
    teacher = {}
    # set numpy random seed
    np.random.seed(1)

    # parse argument
    params = parse_args('train')
    print('--- Enhanced Born-Again Network training: {} ---\n'.format(params.name))
    print(params)

    # output and tensorboard dir
    params.tf_dir = '%s/log/%s' % (params.save_dir, params.name)
    params.checkpoint_dir = '%s/checkpoints/%s' % (params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- prepare dataloader ---')
    print('train with multiple seen domains (unseen domain: {})'.format(params.testset))
    datasets = ['miniImagenet', 'cars', 'places', 'cub', 'plantae']
    datasets.remove(params.testset)
    params.dataset = datasets

    # model
    print('\n--- build BAN model ---')
    if 'Conv' in params.model:
        image_size = 84
    else:
        image_size = 224
    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)

    base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
    test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
    val_file = os.path.join(params.data_dir, 'miniImagenet', 'val.json')
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    # ---- init teacher model structure ---- #
    if params.method == 'matchingnet':
        for dataset in datasets:
            teacher_model = MatchingNet(model_dict[params.teacher], teacher=None,
                                        **train_few_shot_params).cuda()
            # teacher[dataset] = simple_load_model(params, dataset, teacher_model)
            teacher[dataset] = simple_load_teacher(params, dataset, teacher_model)
    # ---- end teacher structure initialize ---- #
    model = MetaDistill(params, teacher, tf_path=params.tf_dir)
    model.cuda()  # model is currently a container

    # resume training
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    model.model.feature.load_state_dict(
        load_warmup_state('%s/checkpoints/%s' % (params.save_dir, params.warmup), params.method), strict=False)

    # training
    print('\n--- start the training ---')
    train(base_datamgr, datasets, val_loader, model, start_epoch, stop_epoch, params)
