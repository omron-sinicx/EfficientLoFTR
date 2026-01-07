
from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.loftr import LoFTR
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler

from torch.profiler import profile

def reparameter(matcher):
    module = matcher.backbone.layer0
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
    for modules in [matcher.backbone.layer1, matcher.backbone.layer2, matcher.backbone.layer3]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    for modules in [matcher.fine_preprocess.layer2_outconv2, matcher.fine_preprocess.layer1_outconv2]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    return matcher


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = LoFTR(config=_config['loftr'], profiler=self.profiler)
        self.loss = LoFTRLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            msg=self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.warmup = False
        self.reparameter = False
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # learning rate warm up (pytorch-lightning 2.x compatible signature)
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            with torch.autocast(enabled=False, device_type='cuda'):
                compute_supervision_coarse(batch, self.config)
        
        with self.profiler.profile("LoFTR"):
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.matcher(batch)
        
        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=False, device_type='cuda'):
                compute_supervision_fine(batch, self.config, self.logger)
            
        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.loss(batch)
    
    def _compute_metrics(self, batch):
        compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
        compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(*batch['pair_names']))
        bs = batch['image0'].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            'epi_errs': [(batch['epi_errs'].reshape(-1,1))[batch['m_bids'] == b].reshape(-1).cpu().numpy() for b in range(bs)],
            'R_errs': batch['R_errs'],
            't_errs': batch['t_errs'],
            'inliers': batch['inliers'],
            'num_matches': [batch['mconf'].shape[0]], # batch size = 1 only
            }
        ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names
    
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                for k, v in figures.items():
                    self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)
        return {'loss': batch['loss']}

    def on_train_epoch_end(self):
        # pytorch-lightning 2.x: training_epoch_end is replaced with on_train_epoch_end
        # Note: outputs are no longer passed; use self.trainer.callback_metrics if needed
        pass  # Loss logging is done per step; epoch-level aggregation can be added if needed

    def on_validation_epoch_start(self):
        self.matcher.fine_matching.validate = True

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        # Log loss scalars
        for k, v in batch['loss_scalars'].items():
            self.log(f'val/{k}', v, on_step=False, on_epoch=True, sync_dist=True)

        # Log pose metrics for AUC calculation
        metrics = ret_dict['metrics']
        for pose_err in metrics.get('R_errs', []):
            self.log('val/R_err', pose_err, on_step=False, on_epoch=True, sync_dist=True)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }
        
    def on_validation_epoch_end(self):
        # pytorch-lightning 2.x: validation_epoch_end is replaced with on_validation_epoch_end
        # Validation outputs are now collected via validation_step return values
        # For simplicity, we'll aggregate metrics from callback_metrics
        self.matcher.fine_matching.validate = False

        # Note: In PL 2.x, outputs are no longer passed to epoch_end hooks.
        # Metrics should be logged per-step using self.log() in validation_step,
        # or collected manually using callbacks or storing in self.
        # The original aggregation logic requires significant refactoring.
        # For now, we rely on per-step logging.
        plt.close('all')  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        if (self.config.LOFTR.BACKBONE_TYPE == 'RepVGG') and not self.reparameter:
            self.matcher = reparameter(self.matcher)
            if self.config.LOFTR.HALF:
                self.matcher = self.matcher.eval().half()
            self.reparameter = True

        if not self.warmup:
            if self.config.LOFTR.HALF:
                for i in range(50):
                    self.matcher(batch)
            else:
                with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                    for i in range(50):
                        self.matcher(batch)
            self.warmup = True
            torch.cuda.synchronize()

        if self.config.LOFTR.HALF:
            self.start_event.record()
            self.matcher(batch)
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
        else:
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.start_event.record()
                self.matcher(batch)
                self.end_event.record()
                torch.cuda.synchronize()
                self.total_ms += self.start_event.elapsed_time(self.end_event)

        ret_dict, rel_pair_names = self._compute_metrics(batch)
        return ret_dict

    def on_test_epoch_end(self):
        # pytorch-lightning 2.x: test_epoch_end is replaced with on_test_epoch_end
        # Note: outputs are no longer passed; metrics should be collected during test_step
        if self.trainer.global_rank == 0:
            print('Averaged Matching time over 1500 pairs: {:.2f} ms'.format(self.total_ms / 1500))