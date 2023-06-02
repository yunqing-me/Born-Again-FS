import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from methods import matchingnet
from methods.backbone import model_dict
from methods import backbone
from tensorboardX import SummaryWriter


class MetaDistill(nn.Module):
    def __init__(self, params, teacher=None, tf_path=None, change_way=True):
        super(MetaDistill, self).__init__()
        # tf writer
        self.params = params
        self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

        # enabling maml training
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.ResNet.maml = True
        if params.method == 'matchingnet':
            backbone.LSTMCell.maml = True
            model = matchingnet.MatchingNet(model_dict[params.model], teacher=teacher,
                                            tf_path=params.tf_dir, **train_few_shot_params)
        else:
            raise ValueError('Unknown method')
        self.model = model
        print('train with {} framework'.format(params.method))
        # get learnable temperature
        self.temperature = backbone.LearnableTemperature(train_few_shot_params).cuda()
        for n, p in self.temperature.named_parameters():
            self.t_params = p
        # params and optimizers
        student_params, teacher_params_dict = self.split_model_parameters_distill()
        self.model_optim = torch.optim.Adam(student_params)
        self.teacher_optim = {}
        for dataset in params.dataset:
            self.teacher_optim[dataset] = torch.optim.Adam(teacher_params_dict[dataset])
        self.t_optim = torch.optim.Adam(self.temperature.parameters())

        # total epochs
        self.total_epoch = params.stop_epoch

    # split teacher and student
    def split_model_parameters_distill(self):
        student_params = []
        teacher_params = {}
        for n, p in self.model.named_parameters():
            m = n.split('.')
            if 'teacher' in m[0]:
                pass
            else:
                student_params.append(p)
        for dataset in self.params.dataset:
            if dataset == 'miniImagenet':
                teacher_params[dataset] = self.model.teacher_miniImagenet.parameters()
            if dataset == 'cub':
                teacher_params[dataset] = self.model.teacher_cub.parameters()
            if dataset == 'cars':
                teacher_params[dataset] = self.model.teacher_cars.parameters()
            if dataset == 'places':
                teacher_params[dataset] = self.model.teacher_places.parameters()
            if dataset == 'plantae':
                teacher_params[dataset] = self.model.teacher_plantae.parameters()
        return student_params, teacher_params

    # train model use maml-like style
    def trainall_loop(self, epoch, ps_set, pu_set, ps_loader, pu_loader, total_it):
        """
        :description: 1. init settings, parameters.
                      2. compute loss/gradients in inner/outer loop
                      3. update parameters. update model on ps-tasks, update temperature on pu-tasks
                      #: in real tests, we no longer have teacher model,
        :return: model
        """
        torch.cuda.empty_cache()
        print_freq = len(ps_loader) / 10
        avg_cross_entropy_loss_inner = 0.
        avg_distill_loss_inner = 0.
        avg_cross_entropy_loss_outer = 0.
        # avg_distill_loss_outer = 0.
        # only optimize teacher model from ps_set to imitate the real recognition
        if ps_set == 'miniImagenet':
            self.model.teacher_ps = self.model.teacher_miniImagenet
        elif ps_set == 'cub':
            self.model.teacher_ps = self.model.teacher_cub
        elif ps_set == 'cars':
            self.model.teacher_ps = self.model.teacher_cars
        elif ps_set == 'places':
            self.model.teacher_ps = self.model.teacher_places
        elif ps_set == 'plantae':
            self.model.teacher_ps = self.model.teacher_plantae
        
        # get the soft knowledge from teacher model mismatched domain
        if pu_set == 'miniImagenet':
            self.model.teacher_pu = self.model.teacher_miniImagenet
        elif pu_set == 'cub':
            self.model.teacher_pu = self.model.teacher_cub
        elif pu_set == 'cars':
            self.model.teacher_pu = self.model.teacher_cars
        elif pu_set == 'places':
            self.model.teacher_pu = self.model.teacher_places
        elif pu_set == 'plantae':
            self.model.teacher_pu = self.model.teacher_plantae

        # start training process
        for i, ((x, _), (x_new, _)) in enumerate(zip(ps_loader, pu_loader)):
            # clear fast weight for maml update
            self.model.train()
            for weight in self.split_model_parameters_distill()[0]:
                weight.fast = None
            # settings on student/teacher model
            self.model.n_query = x.size(1) - self.model.n_support
            self.model.teacher_ps.n_query = self.model.n_query
            self.model.teacher_pu.n_query = self.model.n_query
            if self.model.change_way:
                self.model.n_way = x.size(0)
                self.model.teacher_ps.n_way = self.model.n_way
                self.model.teacher_pu.n_way = self.model.n_way

            # -------------------- update model on domain A, ps tasks --------------------- #
            logits_student = self.model.set_forward(x, is_feature=False)  # logits p
            if self.model.teacher_ps is not None:
                logits_teacher = self.model.teacher_ps.set_forward(x, is_feature=False)  # logits q
                l_js_inner, l_ce_inner = self.distillation_loss_multi(logits_student, logits_teacher)
                model_loss_inner = l_js_inner * 0.8 + l_ce_inner * 1  # \alpha and (1-\alpha)
            else:
                l_js_inner = 0.
                _, l_ce_inner = self.model.set_forward_loss(x)
                model_loss_inner = l_ce_inner

            # --------------- get the outputs from the student and mismatched teacher ---------------- #
            if self.model.teacher_pu is not None:
                logits_teacher_pu = self.model.teacher_pu.set_forward(x, is_feature=False)  # logits q
                l_js_inner, _ = self.distillation_loss_single(logits_student, logits_teacher_pu, T=4)
                model_loss_inner += l_js_inner * 0.5
            else:
                pass

            # Compute inner loop gradients
            meta_grad = torch.autograd.grad(model_loss_inner,
                                            self.split_model_parameters_distill()[0], retain_graph=True)
            for k, weight in enumerate(self.split_model_parameters_distill()[0]):
                weight.fast = weight - self.model_optim.param_groups[0]['lr'] * meta_grad[k]
            meta_grad = [g.detach() for g in meta_grad]

            teacher_grad_ps = torch.autograd.grad(model_loss_inner,
                                                  self.split_model_parameters_distill()[1][ps_set], retain_graph=True)
            teacher_grad_ps = [g.detach() for g in teacher_grad_ps]
            torch.cuda.empty_cache()
            # --------------- update T on domain B, pu tasks ---------------- #
            self.model.eval()
            _, model_loss_outer = self.model.set_forward_loss(x_new)

            # ---- update model ---- #
            self.model_optim.zero_grad()
            self.teacher_optim[ps_set].zero_grad()
            for k, weight in enumerate(self.split_model_parameters_distill()[0]):
                weight.grad = meta_grad[k] if weight.grad is None else weight.grad + meta_grad[k]
            for k, weight in enumerate(self.split_model_parameters_distill()[1][ps_set]):
                weight.grad = teacher_grad_ps[k] if weight.grad is None else weight.grad + teacher_grad_ps[k]

            self.model_optim.step()
            self.teacher_optim[ps_set].step()
            torch.cuda.empty_cache()
            # ---- update T  ---- #
            self.t_optim.zero_grad()
            model_loss_outer.backward()
            self.t_optim.step()

            # ---- store statistic information ---- #
            avg_cross_entropy_loss_inner += l_ce_inner.item()
            avg_distill_loss_inner += l_js_inner.item()
            avg_cross_entropy_loss_outer += model_loss_outer.item()
            torch.cuda.empty_cache()
            if (i + 1) % print_freq == 0:
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | ce_loss_inner {:f} | ban_loss_inner {:f}'.format(
                    epoch + 1, self.total_epoch, i + 1, len(ps_loader),
                    avg_cross_entropy_loss_inner / float(i + 1), avg_distill_loss_inner / float(i + 1)))
            if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar('Inner_cross_entropy_loss', l_ce_inner.item(), total_it + 1)
                self.tf_writer.add_scalar('Inner_ban_loss', l_js_inner.item(), total_it + 1)
                self.tf_writer.add_scalar('Outer_cross_entropy_loss', model_loss_outer.item(), total_it + 1)
                self.tf_writer.add_scalar('Meta-Learned Temperature', self.t_params.item(), total_it + 1)
            total_it += 1

        return total_it

    def distillation_loss_multi(self, student_logits, teacher_logits):
        labels = torch.from_numpy(np.repeat(range(self.model.n_way), self.model.n_query)).cuda()
        # simple symmetric kl-div
        l_kl_1 = self.temperature.forward(student_logits, teacher_logits, log=True)
        l_kl_2 = self.temperature.forward(student_logits, teacher_logits, log=False)
        l_js = 0.5 * (l_kl_1+l_kl_2)
        l_ce = F.cross_entropy(student_logits, labels)

        return l_js, l_ce

    def distillation_loss_single(self, student_logits, teacher_logits, T):
        labels = torch.from_numpy(np.repeat(range(self.model.n_way), self.model.n_query)).cuda()
        p1 = F.log_softmax(student_logits / T, dim=1)
        q1 = F.softmax(teacher_logits / T, dim=1)
        p2 = F.softmax(student_logits / T, dim=1)
        q2 = F.log_softmax(teacher_logits / T, dim=1)

        l_kl_1 = F.kl_div(p1, q1, reduction='batchmean') * (T ** 2)
        l_kl_2 = F.kl_div(q2, p2, reduction='batchmean') * (T ** 2)
        l_js = 0.5 * (l_kl_1+l_kl_2)
        l_ce = F.cross_entropy(student_logits, labels)

        return l_js, l_ce

    def test_loop(self, test_loader, record=None):
        self.model.eval()
        for weight in self.model.parameters():
            weight.fast = None
        return self.model.test_loop(test_loader, record)

    def cuda(self):
        self.model.cuda()
        try:
            self.aux_classifier.cuda()
        except:
            pass

    def reset(self, warmUpState=None):

        # reset feature
        if warmUpState is not None:
            self.model.feature.load_state_dict(warmUpState, strict=False)
            print('reset feature success!')

        # reset other module
        self.model.reset_modules()
        self.model.cuda()

        # reset optimizer
        self.model_optim = torch.optim.Adam(self.split_model_parameters()[0])
        return

    # save function
    def save(self, filename, epoch):
        state = {'epoch': epoch,
                 'model_state': self.model.state_dict(),
                 'model_optim_state': self.model_optim.state_dict()}
        torch.save(state, filename)

    # load function
    def resume(self, filename):
        state = torch.load(filename)
        keys = list(state['model_state'].keys())
        for key in keys:
            m = key.split('.')
            if m[0] == 'teacher_base_set':
                state['model_state'].pop(key)
        self.model.load_state_dict(state['model_state'])
        # self.aux_classifier.load_state_dict(state['aux_classifier_state'])
        self.model_optim.load_state_dict(state['model_optim_state'])
        return state['epoch'] + 1
