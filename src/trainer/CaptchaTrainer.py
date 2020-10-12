import numpy as np
import torch
import os
import torch.nn as nn
from torchvision.utils import make_grid
import torch.nn.functional as F
from tqdm import tqdm
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import model as module_arch


class CaptchaTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, config, data_loader,
                 valid_data_loader=None, len_epoch=None):
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = config.init_obj('lr_scheduler', module_arch.lr_entry, optimizer)
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        # self.train_metrics.reset()
        tbar = tqdm(self.data_loader)
        for batch_idx, (imgid, data, target, input_lengths, target_lengths) in enumerate(tbar):
            data, target = data.to(self.device), target.to(self.device)
            # if batch_size < 2:
            #     continue
            self.optimizer.zero_grad()
            output = self.model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = self.criterion(output_log_softmax, target, input_lengths, target_lengths)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = dict()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        tbar = tqdm(self.valid_data_loader)
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (imgid, data, target, input_lengths, target_lengths) in enumerate(tbar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output_log_softmax = F.log_softmax(output, dim=-1)
                loss = self.criterion(output_log_softmax, target, input_lengths, target_lengths)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (batch_idx + 1)))
                
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', test_loss / (batch_idx + 1))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(self, output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        # Fast test during the training
        result = self.valid_metrics.result()

        return result

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def test(self):
        total_loss = 0.0
        total_metrics = dict()

        if self.do_validation:
            dataloader = self.valid_data_loader
        else:
            dataloader = self.data_loader

        with torch.no_grad():
            for batch_idx, (imgid, data, target, input_lengths, target_lengths) in enumerate(tbar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output_log_softmax = F.log_softmax(output, dim=-1)
                loss = self.criterion(output_log_softmax, target, input_lengths, target_lengths)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (batch_idx + 1)))

                

                val_log = self.valid_metrics.result()
                total_metrics.update(**{'val_' + k: v for k, v in val_log.items()})

        return total_loss, total_metrics
