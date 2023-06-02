import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support,
                 teacher=None, flatten=True, leakyrelu=False, tf_path=None, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        try:
            self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
        except:
            self.feature    = model_func()
        # self.feature = model_func(flatten=flatten, leakyrelu=leakyrelu)
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None
        # specify teacher model
        if teacher is not None:
            try:
                assert isinstance(teacher, dict)
                for dataset in list(teacher.keys()):
                    if dataset == 'miniImagenet':
                        self.teacher_miniImagenet = teacher[dataset]
                    if dataset == 'cub':
                        self.teacher_cub = teacher[dataset]
                    if dataset == 'cars':
                        self.teacher_cars = teacher[dataset]
                    if dataset == 'places':
                        self.teacher_places = teacher[dataset]
                    if dataset == 'plantae':
                        self.teacher_plantae = teacher[dataset]
                print("Initialization on multi-domain finished")
            except:
                self.teacher = teacher
        print("Initialization on single domain finished")

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = x.cuda()
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):
        scores, loss = self.set_forward_loss(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query), loss.item() * len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, total_it):
        print_freq = len(train_loader) // 10
        avg_loss_ce = 0
        avg_loss_js = 0
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            self.teacher.n_query = self.n_query
            if self.change_way:
                self.n_way = x.size(0)
                self.teacher.n_way = self.n_way
            optimizer.zero_grad()
            # soft logits
            logits_student = self.set_forward(x, is_feature=False)  # logits p
            if self.teacher is not None:
                logits_teacher = self.teacher.set_forward(x, is_feature=False)  # logits q
                l_js, l_ce = self.distillation_loss(logits_student, logits_teacher, T=4)
                loss = l_js * 0.8 + l_ce * 1  # \alpha and (1-\alpha)
                avg_loss_ce = avg_loss_ce + l_ce.item()
                avg_loss_js = avg_loss_js + l_js.item()
            else:
                l_js = 0.
                l_ce = self.set_forward_loss(x)
                loss = l_ce
                avg_loss_ce = avg_loss_ce + l_ce.item()
                avg_loss_js = avg_loss_js + l_js

            loss.backward()
            optimizer.step()

            if (i + 1) % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | ce-loss {:f} | ban-loss {:f}'.format(
                      epoch + 1, i + 1, len(train_loader), avg_loss_ce / float(i + 1), avg_loss_js / float(i + 1)))
            if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar(self.method + '/query_ce-loss', l_ce.item(), total_it + 1)
                self.tf_writer.add_scalar(self.method + '/query_ban-loss', l_js.item(), total_it + 1)
            total_it += 1
        return total_it

    def distillation_loss(self, student_logits, teacher_logits, T):
        labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
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
        loss = 0.
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this, loss_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)
            loss += loss_this
            count += count_this
            if i > 100:
                break

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('--- %d Loss = %.6f ---' % (iter_num, loss / count))
        print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean
