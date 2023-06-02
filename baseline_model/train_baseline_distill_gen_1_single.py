import numpy as np
import torch
import torch.optim
import os

from methods import backbone
from methods.backbone import model_dict
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain_distill_single import BaselineTrain 
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.gnnnet import GnnNet
from options import parse_args, get_resume_file, load_warmup_state
from collections import OrderedDict


def train(base_loader, model, teacher_model, start_epoch, stop_epoch, params):

  # get optimizer and checkpoint path
  optimizer = torch.optim.Adam(model.parameters())
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)
  total_it = 0

  # pre-train on multiple source domains
  for epoch in range(start_epoch,stop_epoch):
    model.train()
    total_it = model.train_loop(epoch, base_loader, optimizer, total_it, teacher_model) #model are called by reference, no need to return

    # save checkpoints 
    if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

  return model


# --- main function ---
if __name__=='__main__':
  # set numpy random seed
  np.random.seed(10)
  
  # parser argument
  params = parse_args('train')

  # # for debug
  # params.method  = 'baseline'
  # params.dataset = 'miniImagenet'

  print('--- baseline training: {} ---\n'.format(params.name))
  print(params)

  # output and tensorboard dir
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # model
  print('\n--- build model ---')
  if 'Conv' in params.model:
    image_size = 84
  else:
    image_size = 224
    

  # ---- Step 1. initialize multiple source domains, will sample separately ---- #
  # dataloader
  print('\n--- prepare dataloader ---')
  if params.dataset == 'multi':
    raise ValueError('require single source domain')

  if params.method in ['baseline', 'baseline++'] :
    print('pre-training the feature encoder {} using method {} with linear classifier'.format(params.model, params.method))
    base_datamgr    = SimpleDataManager(image_size, batch_size=16)
    
    base_loader = base_datamgr.get_data_loader(os.path.join(params.data_dir, params.dataset, 'base.json') , aug=params.train_aug)
    
    # define models
    if params.method == 'baseline':
      model         = BaselineTrain(model_dict[params.model], params.num_classes)
      teacher_model = BaselineTrain(model_dict[params.model], params.num_classes)
    
    elif params.method == 'baseline++':
      model         = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')
      teacher_model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')
  else:
    raise ValueError('Unknown method')
  model = model.cuda()
  teacher_model = teacher_model.cuda()
  
  # load model
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch


  # --- conventional born-again networks with multi-head training --- #
  
  # step 1. load the student/teacher model
  if params.gen == 1:
    params.warmup = f'baseline'
    model_state = torch.load(f'{params.save_dir}/checkpoints/{params.warmup}/399.tar')
  elif params.gen >= 2:
    params.warmup = f'baseline_gen_{params.gen-1}'
    model_state = torch.load(f'{params.save_dir}/checkpoints/{params.warmup}/349.tar')

  # load weights
  model.load_state_dict(model_state['state'], strict=True)
  teacher_model.load_state_dict(model_state['state'], strict=True)
  teacher_model.eval()

  # step 2. born-again training
  print('\n--- start the training ---')
  model = train(base_loader, model, teacher_model, start_epoch, stop_epoch, params)
