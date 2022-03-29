import sys
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import time
import copy
from geoopt.manifolds.stereographic import math as math1
import torch
import cupy
import math
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn


import utils as U


class Trainer:
    def __init__(self, model, optimizer, train_iter, val_iter, opt):
        self.model = model
        self.criterion = U.kl_divergence if opt.BC else F.softmax_cross_entropy
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.opt = opt
        self.n_batches = (len(train_iter.dataset) - 1) // opt.batchSize + 1
        self.start_time = time.time()

    def get_loss_and_acc(self, inputs1, targets1, targets2=None, ratio=None):
        output = self.model(inputs1)
        loss = self.criterion(output, targets1)

        if targets2 is not None:
            loss = loss * ratio + self.criterion(output, targets2) * (1 - ratio)
        loss = F.mean(loss)

        acc = F.accuracy(output, targets1)

        return loss, acc

    def train(self, epoch):
        self.optimizer.lr = self.lr_schedule(epoch)
        train_loss = 0
        train_acc = 0

        for i, batch in enumerate(self.train_iter):

            x_array, t_array, l_array = chainer.dataset.concat_examples(batch)

            x = chainer.Variable(cuda.to_gpu(x_array[:, None, None, :]))
            t = chainer.Variable(cuda.to_gpu(t_array))
            l = chainer.Variable(cuda.to_gpu(l_array))

            #################################################################
            # Adding the ssmix function
            splitted = self.split_labeled_batch(x, t, l)
            x_left, t_left, l_left, x_right, t_right, l_right = splitted

            if x_left is None:  # Odd length batch
                continue

            mix_input1, ratio_left = self.augment(
                x_left, x_right, t_left, t_right, l_left, l_right
            )
            mix_input2, ratio_right = self.augment(
                x_right, x_left, t_right, t_left, l_left, l_right
            )

            ###################################################################

            self.model.cleargrads()

            loss_list = []
            acc_list = []

            loss1, acc1 = self.get_loss_and_acc(x_left, t_left)
            loss2, acc2 = self.get_loss_and_acc(x_right, t_right)
            loss3, acc3 = self.get_loss_and_acc(mix_input1, t_left, t_right, ratio_left)
            loss4, acc4 = self.get_loss_and_acc(
                mix_input2, t_right, t_left, ratio_right
            )

            loss_list = F.stack([loss1, loss2, loss3, loss4])
            acc_list = F.stack([acc1, acc2, acc3, acc4])

            loss = F.mean(loss_list)
            acc = F.mean(acc_list)

            loss.backward()
            self.optimizer.update()
            train_loss += float(loss.data) * len(t_array.data)
            train_acc += float(acc.data) * len(t_array.data)

            elapsed_time = time.time() - self.start_time
            progress = (
                (self.n_batches * (epoch - 1) + i + 1)
                * 1.0
                / (self.n_batches * self.opt.nEpochs)
            )
            eta = elapsed_time / progress - elapsed_time

            line = "* Epoch: {}/{} ({}/{}) | Train: LR {} | Time: {} (ETA: {})".format(
                epoch,
                self.opt.nEpochs,
                i + 1,
                self.n_batches,
                self.optimizer.lr,
                U.to_hms(elapsed_time),
                U.to_hms(eta),
            )
            sys.stderr.write("\r\033[K" + line)
            sys.stderr.flush()

        self.train_iter.reset()
        train_loss /= len(self.train_iter.dataset)
        train_top1 = 100 * (1 - train_acc / len(self.train_iter.dataset))

        return train_loss, train_top1
    
    def val_mixup(self):
        self.model.train = False
        acc = 0

        for i, batch in enumerate(self.val_iter):

            x_array, t_array, l_array = chainer.dataset.concat_examples(batch)

            with chainer.using_config("volatile", True):
                x = chainer.Variable(cuda.to_gpu(x_array[:, None, None, :]))
                t = chainer.Variable(cuda.to_gpu(t_array))
                l = chainer.Variable(cuda.to_gpu(l_array))

            #################################################################
            # Adding the ssmix function
            splitted = self.split_labeled_batch(x, t, l)
            x_left, t_left, l_left, x_right, t_right, l_right = splitted

            if x_left is None:  # Odd length batch
                continue

            mix_input1, ratio_left = self.augment(
                x_left, x_right, t_left, t_right, l_left, l_right
            )
            mix_input2, ratio_right = self.augment(
                x_right, x_left, t_right, t_left, l_left, l_right
            )

            ###################################################################

            acc_list = []

            _, acc1 = self.get_loss_and_acc(x_left, t_left)
            _, acc2 = self.get_loss_and_acc(x_right, t_right)
            _, acc3 = self.get_loss_and_acc(mix_input1, t_left, t_right, ratio_left)
            _, acc4 = self.get_loss_and_acc(
                mix_input2, t_right, t_left, ratio_right
            )
            
            acc_list = F.stack([acc1, acc2, acc3, acc4])

            acc = F.mean(acc_list)

            acc += float(acc.data) * len(t_array.data)

            elapsed_time = time.time() - self.start_time
            progress = (
                (self.n_batches + i + 1)
                * 1.0
                / (self.n_batches)
            )
            eta = elapsed_time / progress - elapsed_time

            line = "* ({}/{}) | Time: {} (ETA: {})".format(
                i + 1,
                self.n_batches,
                U.to_hms(elapsed_time),
                U.to_hms(eta),
            )
            sys.stderr.write("\r\033[K" + line)
            sys.stderr.flush()

        self.val_iter.reset()
        self.model.train = True
        val_top1 = 100 * (1 - acc / len(self.val_iter.dataset))

        return val_top1

    def val(self):
        self.model.train = False
        val_acc = 0
        for batch in self.val_iter:
            x_array, t_array, *_ = chainer.dataset.concat_examples(batch)
            if self.opt.nCrops > 1:
                x_array = x_array.reshape(
                    (x_array.shape[0] * self.opt.nCrops, x_array.shape[2])
                )

            with chainer.using_config("volatile", True):
                x = chainer.Variable(cuda.to_gpu(x_array[:, None, None, :]))
                t = chainer.Variable(cuda.to_gpu(t_array))

            y = F.softmax(self.model(x))
            y = F.reshape(
                y, (y.shape[0] // self.opt.nCrops, self.opt.nCrops, y.shape[1])
            )
            y = F.mean(y, axis=1)
            acc = F.accuracy(y, t)
            val_acc += float(acc.data) * len(t.data)

        self.val_iter.reset()
        self.model.train = True
        val_top1 = 100 * (1 - val_acc / len(self.val_iter.dataset))

        return val_top1

    def lr_schedule(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule])
        decay = sum(epoch > divide_epoch)
        if epoch <= self.opt.warmup:
            decay = 1

        return self.opt.LR * np.power(0.1, decay)

    def split_labeled_batch(self, inputs, labels, lens):
        if len(inputs) % 2 != 0:
            # Skip any odd numbered batch
            return None, None, None, None, None, None

        inputs_left, inputs_right = F.split_axis(
            inputs, axis=0, indices_or_sections=2, force_tuple=True
        )
        labels_left, labels_right = F.split_axis(
            labels, axis=0, indices_or_sections=2, force_tuple=True
        )
        lens_left, lens_right = F.split_axis(
            lens, axis=0, indices_or_sections=2, force_tuple=True
        )

        return (
            inputs_left,
            labels_left,
            lens_left,
            inputs_right,
            labels_right,
            lens_right,
        )

    def augment(self, inputs1, inputs2, target1, target2, l_left, l_right):
        return self.ssmix(inputs1, inputs2, target1, target2, l_left, l_right)

    def ssmix(self, inputs1, inputs2, target1, target2, l_left, l_right):
        inputs_aug = copy.deepcopy(inputs1)

        args = {
            "model": self.model,
            "optimizer": self.optimizer,
            "criterion": U.kl_divergence if self.opt.BC else F.softmax_cross_entropy,
        }

        saliency1 = U.get_saliency(args, inputs_aug, target1)
        saliency2 = U.get_saliency(args, copy.deepcopy(inputs2), target2)

        batch_size = len(inputs1) * 2
        mix_size = max(int(self.opt.inputLength * (self.opt.ss_winsize / 100.0)), 1)

        ratio = cuda.to_gpu(np.ones((batch_size,)))

        # inputs_mixed=chainer.Variable(cuda.to_gpu(np.array([],dtype='float32')))
        inputs_mixed = []

        for i in range(batch_size // 2):

            l1 = l_left[i].data
            l2 = l_right[i].data
            saliency1_eg = saliency1[i]  # (1,1,60k) - (1,1,1,60k)
            saliency2_eg = saliency2[i]

            mix_ratio = 1 - (mix_size / (self.opt.inputLength))

            stride_len = int(self.opt.stride_length * self.opt.fs)

            start_idx1 = 0
            start_idx2 = 0

            if self.opt.ignorePad:
                nopad_start1 = (
                    0
                    if l1 >= self.opt.inputLength
                    else (self.opt.inputLength - l1) // 2
                )
                nopad_start2 = (
                    0
                    if l2 >= self.opt.inputLength
                    else (self.opt.inputLength - l2) // 2
                )
                nopad_end1 = nopad_start1 + l1
                nopad_end2 = nopad_start2 + l2

                saliency1_eg = saliency1_eg[:, :, nopad_start1:nopad_end1]
                saliency2_eg = saliency2_eg[:, :, nopad_start2:nopad_end2]

                start_idx1 = nopad_start1
                start_idx2 = nopad_start2

                mix_ratio = 1 - mix_size / l1

            if self.opt.hyp_mean == 1:

                # Hyperbolic sectional curvature
                c = torch.tensor([1.0])
                c = c.to(device="cuda")

                saliency1_eg = F.squeeze(F.squeeze(saliency1_eg, axis=0), axis=0)
                saliency2_eg = F.squeeze(F.squeeze(saliency2_eg, axis=0), axis=0)

                saliency1_eg = cupy.asnumpy(saliency1_eg.data)
                saliency2_eg = cupy.asnumpy(saliency2_eg.data)

                saliency1_eg = torch.from_numpy(saliency1_eg)
                saliency2_eg = torch.from_numpy(saliency2_eg)

                saliency1_eg = torch.unsqueeze(saliency1_eg, 1)
                saliency2_eg = torch.unsqueeze(saliency2_eg, 1)

                saliency1_eg = saliency1_eg.to(device="cuda")
                saliency2_eg = saliency2_eg.to(device="cuda")

                saliency1_proj = math1.project(saliency1_eg, k=c)
                saliency1_proj = math1.expmap0(saliency1_proj, k=c)

                saliency2_proj = math1.project(saliency2_eg, k=c)
                saliency2_proj = math1.expmap0(saliency2_proj, k=c)

                span_saliency1 = []
                span_saliency2 = []

                # Finding saliency of various spans
                if self.opt.ignorePad:
                    end_span_start1 = start_idx1 + l1 - mix_size
                    end_span_start2 = start_idx2 + l2 - mix_size

                    for j in range(0, end_span_start1, stride_len):
                        span_saliency1.append(saliency1_proj[j : j + mix_size])

                    for j in range(0, end_span_start2, stride_len):
                        span_saliency2.append(saliency2_proj[j : j + mix_size])
                else:
                    end_span_start = self.opt.inputLength - mix_size

                    for j in range(0, end_span_start, stride_len):
                        span_saliency1.append(saliency1_proj[j : j + mix_size])
                        span_saliency2.append(saliency2_proj[j : j + mix_size])

                saliency1_list = []
                saliency2_list = []

                span_saliency1 = torch.stack(span_saliency1)
                span_saliency2 = torch.stack(span_saliency2)

                span_saliency1 = span_saliency1.to(device="cuda")
                span_saliency2 = span_saliency2.to(device="cuda")

                if self.opt.ignorePad:
                    for j in range(0, len(span_saliency1)):
                        mean1 = math1.weighted_midpoint(
                            span_saliency1[j], k=c
                        )  # [13330,1]
                        saliency1_list.append(mean1)

                    for j in range(0, len(span_saliency2)):
                        mean2 = math1.weighted_midpoint(span_saliency2[j], k=c)
                        saliency2_list.append(mean2)
                else:
                    for j in range(0, len(span_saliency1)):
                        mean1 = math1.weighted_midpoint(
                            span_saliency1[j], k=c
                        )  # [13330,1]
                        mean2 = math1.weighted_midpoint(span_saliency2[j], k=c)

                        saliency1_list.append(mean1)
                        saliency2_list.append(mean2)

                saliency1_list = torch.tensor(saliency1_list)
                saliency2_list = torch.tensor(saliency2_list)

                saliency1_list = torch.unsqueeze(saliency1_list, 1)  # [13,1]
                saliency2_list = torch.unsqueeze(saliency2_list, 1)

                saliency1_list = saliency1_list.to(device="cuda")
                saliency2_list = saliency2_list.to(device="cuda")

                saliency1_list_euc = math1.logmap0(saliency1_list, k=c)
                saliency1_list_euc = math1.project(saliency1_list_euc, k=c)

                saliency2_list_euc = math1.logmap0(saliency2_list, k=c)
                saliency2_list_euc = math1.project(saliency2_list_euc, k=c)

                saliency1_list_euc = torch.squeeze(saliency1_list_euc)
                saliency2_list_euc = torch.squeeze(saliency2_list_euc)

                # To find the starting idx of the span
                input1_idx = (
                    int(torch.argmin(saliency1_list_euc).data) * stride_len + start_idx1
                )
                input2_idx = (
                    int(torch.argmax(saliency2_list_euc).data) * stride_len + start_idx2
                )

            else:
                saliency1_pool = F.squeeze(
                    F.squeeze(
                        F.average_pooling_1d(saliency1_eg, mix_size, stride=stride_len),
                        axis=0,
                    ),
                    axis=0,
                )

                saliency2_pool = F.squeeze(
                    F.squeeze(
                        F.average_pooling_1d(saliency2_eg, mix_size, stride=stride_len),
                        axis=0,
                    ),
                    axis=0,
                )

                input1_idx = (
                    int(F.argmin(saliency1_pool).data) * stride_len + start_idx1
                )
                input2_idx = (
                    int(F.argmax(saliency2_pool).data) * stride_len + start_idx2
                )

            left_span = inputs_aug[i, :, :, :input1_idx]
            replace_span = inputs2[i, :, :, input2_idx : input2_idx + mix_size]
            right_span = inputs_aug[i, :, :, input1_idx + mix_size :]

            mixed = F.concat([left_span, replace_span, right_span], axis=2)
            inputs_mixed.append(mixed)

            # inputs_aug[i, :, :, input1_idx:input1_idx + mix_size] = inputs2[
            #     i, :, :, input2_idx:input2_idx + mix_size
            # ]

            ratio[i] = mix_ratio

        inputs_mixed = F.stack(inputs_mixed, axis=0)

        return inputs_mixed, ratio
