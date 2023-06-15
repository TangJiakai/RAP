from time import time
import logging
from evaluate import evaluate_funs

from tqdm import tqdm
from utils import dict2str, early_stopping, get_local_time, get_tensorboard, clear_dir, remove_tensorboard_dir, set_color

import torch
import numpy as np

unrecord_params = ['num_workers', 'data_type', 'beta', 'use_tensorboard', 'use_gpu', 'gpu_id', 'log', 'saved', 'resume_file', 'test', 'train_data_path',
'valid_data_path', 'saved_model_path_pre', 'tensorboard_dir', 'device', 'use_wandb', 'device']


class Trainer(object):
    def __init__(self, config, model):
        self.logger = logging.getLogger()
        self.config = config
        self.model = model
        self.epochs = config['epochs']
        self.optimizer_type = config['optimizer_type']
        self.learning_rate = config['lr']
        self.weight_decay = config['weight_decay']
        self.eval_step = config['eval_step']
        self.stopping_step = config['stopping_step']
        self.saved = config['saved']
        self.device = config['device']
        self.saved_model_path_pre = config['saved_model_path_pre'] if 'saved_model_path_pre' in config else None
        self.tensorboard = None
        self.cur_step = 0
        self.start_epoch = 0
        self.best_valid_score = -np.inf
        self.best_valid_result = None
        self.optimizer = self._build_optimizer()
        self.train_loss_dict = dict()
        self.best_model_file = None

        self.pretrain_step = config['pretrain_step'] if 'pretrain_step' in config else -1

    def _build_optimizer(self):
        optimizer_type = self.optimizer_type
        learning_rate = self.learning_rate
        weight_decay = self.weight_decay

        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def train_early_stopping(self, valid_data, epoch_idx):
        valid_start_time = time()
        valid_score, valid_result = self._valid_epoch(valid_data)
        (
            self.best_valid_score,
            self.cur_step,
            stop_flag,
            update_flag,
        ) = early_stopping(
            valid_score,
            self.best_valid_score,
            self.cur_step,
            max_step=self.stopping_step,
        )
        valid_end_time = time()
        valid_score_output = (
            set_color("epoch %d evaluating", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
            + set_color("valid_score", "blue")
            + ": %f]"
        ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
        valid_result_output = (
            set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
        )
        self.logger.info(valid_score_output)
        self.logger.info(valid_result_output)
        self._add_valid_metrics_to_tensorboard(epoch_idx, valid_result)
        return valid_result, update_flag, stop_flag

    def fit(self, train_dataloader, valid_dataloader):
        if self.tensorboard is None:
            self.tensorboard = get_tensorboard(self.config['tensorboard_dir'] if 'tensorboard_dir' in self.config else None)

        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()

            if hasattr(self.model, 'pretrain_calculate_loss') and self.pretrain_step > epoch_idx:
                train_loss = self._pretrain_epoch(train_dataloader, epoch_idx)
            else:
                train_loss = self._train_epoch(train_dataloader, epoch_idx)

            self.train_loss_dict[epoch_idx] = (sum(train_loss) if isinstance(train_loss, tuple) else train_loss)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if hasattr(self.model, 'pretrain_calculate_loss') and self.pretrain_step > epoch_idx:
                if (epoch_idx + 1) % self.eval_step == 0:
                    valid_start_time = time()
                    valid_score, valid_result = self._valid_epoch(valid_dataloader)
                    valid_end_time = time()
                    valid_score_output = (
                        set_color("epoch %d evaluating", "green")
                        + " ["
                        + set_color("time", "blue")
                        + ": %.2fs, "
                        + set_color("valid_score", "blue")
                        + ": %f]"
                    ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                    valid_result_output = (
                        set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                    )
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self._add_valid_metrics_to_tensorboard(epoch_idx, valid_result)
                continue

            if (self.eval_step <= 0) and self.saved:
                if epoch_idx == self.epochs - 1:
                    valid_result, update_flag, stop_flag = self.train_early_stopping(valid_dataloader, epoch_idx)
                    self._save_checkpoint(epoch_idx)
                continue

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_result, update_flag, stop_flag = self.train_early_stopping(valid_dataloader, epoch_idx)
                if update_flag:
                    if self.saved:
                        self._save_checkpoint(epoch_idx)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result

    def _train_epoch(self, train_dataloader, epoch_idx):
        self.model.train()
        loss_func = self.model.calculate_loss
        total_loss = None
        iter_data = tqdm(train_dataloader, total=len(train_dataloader), ncols=100, desc=set_color(f"Train {epoch_idx:>5}", "pink"))

        for batch_data in iter_data:
            batch_data = [x.to(self.device) for x in batch_data]
            losses = loss_func(batch_data)
            
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self.optimizer.zero_grad()
            loss.backward()
            self._check_nan(loss)
            self.optimizer.step()
        return total_loss
    
    def _pretrain_epoch(self, train_dataloader, epoch_idx):
        self.model.train()
        loss_func = self.model.pretrain_calculate_loss
        total_loss = None
        iter_data = tqdm(train_dataloader, total=len(train_dataloader), ncols=100, desc=set_color(f"Train {epoch_idx:>5}", "pink"))

        for batch_data in iter_data:
            batch_data = [x.to(self.device) for x in batch_data]
            losses = loss_func(batch_data)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self.optimizer.zero_grad()
            loss.backward()
            self._check_nan(loss)
            self.optimizer.step()
        return total_loss

    def _valid_epoch(self, valid_dataloader):
        if not valid_dataloader:
            return
        
        valid_score, valid_result = self.evaluate(valid_dataloader, load_best_model=False)
        return valid_score, valid_result

    @torch.no_grad()
    def evaluate(self, valid_dataloader, load_best_model=True, model_file=None):
        if not valid_dataloader:
            return None

        if load_best_model:
            checkpoint_file = model_file or self.best_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()
        iter_data = tqdm(valid_dataloader, total=len(valid_dataloader), ncols=100, desc=set_color(f"Evaluate   ", "pink"))

        pred_scores = []
        pos_item_ids = []
        history_item_ids = []
        for batch_data in iter_data:
            valid_batch_data = batch_data[0].to(self.device)
            pred_scores.append(self.model.full_rank(valid_batch_data).cpu().clone().detach())
            history_item_ids.extend(batch_data[1])
            pos_item_ids.extend(batch_data[2])
        valid_score, valid_result = evaluate_funs(history_item_ids, pos_item_ids, pred_scores)
        if load_best_model:
            self._add_hparam_to_tensorboard(valid_result)
            self.logger.info('test result:\t' + str(valid_result))
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        clear_dir(self.saved_model_path_pre)
        best_model_file = self.saved_model_path_pre + f"[{self.best_valid_score}]-{epoch}-[{get_local_time()}].pth"
        self.best_model_file = best_model_file
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "best_model_file": self.best_model_file,
            "saved_model_path_pre": self.saved_model_path_pre,
            "state_dict": self.model.state_dict(),
            "other_parameter": self.model.other_parameter(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, best_model_file)
        print(set_color("Saving current", "blue") + f": {best_model_file}")

    def resume_checkpoint(self, resume_file):
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint["epoch"] + 1
        self.cur_step = checkpoint["cur_step"]
        self.best_valid_score = checkpoint["best_valid_score"]
        self.best_model_file = checkpoint["best_model_file"]
        self.saved_model_path_pre = checkpoint["saved_model_path_pre"]
        self.tensorboard = get_tensorboard(self.config['tensorboard_dir'] if 'tensorboard_dir' in self.config else None, purge_step=self.cur_step)

        # load architecture params from checkpoint
        if checkpoint["config"]["model"] != self.config["model"]:
            raise ValueError(
                "Architecture configuration given in config file is different from that of checkpoint. "
                "This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_other_parameter(checkpoint.get("other_parameter"))

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        message_output = "Checkpoint loaded. Resume training from epoch {}".format(
            self.start_epoch
        )
        self.logger.info(message_output)
        
    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        des = None
        if isinstance(losses, tuple):
            des = set_color("train_loss%d", "blue") + ": %.4f"
            train_loss_output += ", ".join(
                des % (idx + 1, loss) for idx, loss in enumerate(losses)
            )
        else:
            des = "%.4f"
            train_loss_output += set_color("train loss", "blue") + ": " + des % losses
        return train_loss_output + "]"

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Train/Loss'):
        if self.tensorboard is None:
            return
            
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + "_" + str(idx), loss, epoch_idx, new_style=True)
            self.tensorboard.add_scalar(tag + "_total", sum(losses), epoch_idx, new_style=True)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx, new_style=True)

    def _add_valid_metrics_to_tensorboard(self, epoch_idx, valid_result):
        if self.tensorboard is None:
            return None

        for key, value in valid_result.items():
            self.tensorboard.add_scalar("Valid/Metric" + "_" + key, value, epoch_idx, new_style=True)
    
    def _add_hparam_to_tensorboard(self, best_valid_result, metric_tag="hparam/test_result_"):
        if self.tensorboard is None:
            return
        remove_tensorboard_dir(self.config['tensorboard_dir'])
        hparam_dict = dict()
        for key, value in self.config.items():
            if key not in unrecord_params:
                hparam_dict[key] = value
        
        for k in hparam_dict:
            if hparam_dict[k] is not None and not isinstance(
                hparam_dict[k], (bool, str, float, int)
            ):
                hparam_dict[k] = str(hparam_dict[k])
        
        self.tensorboard.add_hparams(hparam_dict, {metric_tag + k:v for k, v in best_valid_result.items()})    
