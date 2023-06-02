import numpy as np
import torch
import torch.optim
import os

from methods import backbone
from methods.backbone import model_dict
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain_distill_multi import BaselineTrain 
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.gnnnet import GnnNet
from options import parse_args, get_resume_file, load_warmup_state
from collections import OrderedDict


def train(base_loader, model, teacher_model, start_epoch, stop_epoch, params):

  # get optimizer and checkpoint path
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)
  
  # combine parameters
  feature_params    = []
  for n, p in model.named_parameters():
    if 'feature' in n:
      feature_params.append(p)
  
  classifier_params = []
  for source_name in datasets:
    if source_name == 'miniImagenet':
      classifier_params =  list(model.classifier[source_name].parameters())
    else:
      classifier_params += list(model.classifier[source_name].parameters())
      
  optimizer = torch.optim.Adam(feature_params + classifier_params)
  total_it = 0

  # pre-train on multiple source domains
  for epoch in range(start_epoch,stop_epoch):
    model.train()
    total_it = model.train_loop(epoch, base_loader, optimizer, total_it, teacher_model) #model are called by reference, no need to return

    # save checkpoints 
    if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch': epoch,
                  'feature': model.state_dict(),
                  f'classifier_{datasets[0]}': model.classifier[datasets[0]].state_dict(),
                  f'classifier_{datasets[1]}': model.classifier[datasets[1]].state_dict(),
                  f'classifier_{datasets[2]}': model.classifier[datasets[2]].state_dict(),
                  f'classifier_{datasets[3]}': model.classifier[datasets[3]].state_dict(),
                  }, outfile)

  return model


# --- main function ---
if __name__=='__main__':
  # set numpy random seed
  np.random.seed(10)

  # parser argument
  params = parse_args('train')
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
    print('train with multiple seen domains (unseen domain: {})'.format(params.testset))
    # source dataset name
    datasets = ['miniImagenet', 'cars', 'places', 'cub', 'plantae']
    datasets.remove(params.testset)
  else:
    raise ValueError('require multi source domains')

  base_loader = dict()
  if params.method in ['baseline', 'baseline++'] :
    print('pre-training the feature encoder {} using method {} with multi-head classifier'.format(params.model, params.method))
    base_datamgr    = SimpleDataManager(image_size, batch_size=16)
    
    # wrap data loaders
    for source_name in datasets:
      base_loader[source_name] = base_datamgr.get_data_loader(os.path.join(params.data_dir, source_name, 'base.json') , aug=params.train_aug)
    
    # init multi-head classifiers 
    num_classes_multi = {'miniImagenet': 100,
                         'cub': 200, 
                         'cars': 300,
                         'places': 400,
                         'plantae': 200
                         }
    # define models
    if params.method == 'baseline':
      model         = BaselineTrain(model_dict[params.model], num_classes_multi, datasets=datasets)
      teacher_model = BaselineTrain(model_dict[params.model], num_classes_multi, datasets=datasets)
    elif params.method == 'baseline++':
      model         = BaselineTrain(model_dict[params.model], num_classes_multi, loss_type='dist', datasets=datasets)
      teacher_model = BaselineTrain(model_dict[params.model], num_classes_multi, loss_type='dist', datasets=datasets)
  else:
    raise ValueError('Unknown method')
  model = model.cuda()
  teacher_model = teacher_model.cuda()
  
  for source_name in datasets:
    model.classifier[source_name] = model.classifier[source_name].cuda()
    teacher_model.classifier[source_name] = teacher_model.classifier[source_name].cuda()
    
  # load model
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch


  # --- conventional born-again networks with multi-head training --- #
  
  # step 1. load the student/teacher model
  params.warmup = f'baseline++_seen_multi_unseen_{params.testset}_gen_{params.gen-1}'
  model_state = torch.load(f'{params.save_dir}/checkpoints/{params.warmup}/399.tar')
  
  # load backbone weights
  # model.load_state_dict(model_state['feature'], strict=True)
  teacher_model.load_state_dict(model_state['feature'], strict=True)
  teacher_model.eval()
  
  # load classifier head weights
  for source_name in datasets:
    tmdsad = model.classifier[source_name].state_dict()  
    head_dict = OrderedDict()
    for i, key in enumerate(model_state[f'classifier_{source_name}'].keys()):
      if 'v' in key:   # weights
        head_dict['weight'] = model_state[f'classifier_{source_name}'][key]
      elif 'g' in key: # bias
        head_dict['bias']   = model_state[f'classifier_{source_name}'][key].squeeze()
        
    # model.classifier[source_name].load_state_dict(model_state[f'classifier_{source_name}'], strict=True)
    teacher_model.classifier[source_name].load_state_dict(model_state[f'classifier_{source_name}'], strict=True)  
    # teacher_model.classifier[source_name].load_state_dict(head_dict, strict=True)  
    teacher_model.classifier[source_name].eval()

  # step 2. born-again training
  print('\n--- start the training ---')
  model = train(base_loader, model, teacher_model, start_epoch, stop_epoch, params)
