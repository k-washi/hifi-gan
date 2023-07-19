import itertools
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from pytorch_lightning import LightningModule

from src.avocodo.config import AvocodoConfig
from src.avocodo.avocodo import Avocodo
from src.avocodo.CoMBD import CoMBD
from src.avocodo.SBD import SBD
from src.avocodo.loss import (
    feature_loss,
    generator_loss,
    discriminator_loss,
)
from src.avocodo.pqmf import PQMF
from src.utils.audio import mel_spectrogram_torch


class AvocodoModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        
        # for gan
        self.automatic_optimization = False
        
        h = AvocodoConfig()
        h.update_config(cfg)
        self.cfg = cfg
        self.h: AvocodoConfig = h
        
        self.pqmf_lv2 = PQMF(*h.pqmf_lv2)
        self.pqmf_lv1 = PQMF(*h.pqmf_lv1)
        
        self.model = Avocodo(h)
        self.combd = CoMBD(h.combd, pqmf_list=[self.pqmf_lv2, self.pqmf_lv1])
        self.sbd = SBD(h.sbd)
    
    def configure_optimizers(self):
        lr = self.cfg.ml.optim.learning_rate
        b1, b2 = self.cfg.ml.optim.optimizer_beta1, self.cfg.ml.optim.optimizer_beta2
        opt_g = torch.optim.AdamW(
            self.model.parameters(),
            lr,
            betas=[b1, b2]
        )
        opt_d = torch.optim.AdamW(
            itertools.chain(self.combd.parameters(), self.sbd.parameters()),
            lr,
            betas=[b1, b2]
        )
        
        if self.cfg.ml.scheduler.use:
            if self.cfg.ml.scheduler.name == "ExponentialLR":
                scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=opt_g, gamma=self.cfg.ml.scheduler.gamma
                )
            
            
           
                scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=opt_d, gamma=self.cfg.ml.scheduler.gamma
                )
            else:
                raise ValueError(f"Scheduler {self.cfg.ml.scheduler.name} is not supported.")

            
            scheduler_g = {
                "scheduler": scheduler_g,
                "interval": self.cfg.ml.scheduler.interval,
            }
            scheduler_d = {
                "scheduler": scheduler_d,
                "interval":  self.cfg.ml.scheduler.interval,
            }
            return [opt_g, opt_d], [scheduler_g, scheduler_d]
        return [opt_g, opt_d], []
    
    def forward(self, z):
        return self.model(z)[-1]
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        y, x, y_mel, _ = batch
        optimizer_g, optimizer_d = self.optimizers()

        y = y.unsqueeze(1)
        ys = [
            self.pqmf_lv2.analysis(
                y
            )[:, :self.h.projection_filters[1]],
            self.pqmf_lv1.analysis(
                y
            )[:, :self.h.projection_filters[2]],
            y
        ]
        
        y_g_hats = self.model(x)
        
        ## train generator
        y_du_hat_r, y_du_hat_g, fmap_u_r, fmap_u_g = self.combd(ys, y_g_hats)
        loss_fm_u, losses_fm_u = feature_loss(fmap_u_r, fmap_u_g)
        loss_gen_u, losses_gen_u = generator_loss(y_du_hat_g)
        
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.sbd(y, y_g_hats[-1])
        loss_fm_s, losses_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        
        # L1 Mel-Spectrogram Loss
        y_g_hat_mel = mel_spectrogram_torch(
            y_g_hats[-1].squeeze(1),
            n_fft=self.cfg.dataset.fft_size,
            num_mels=self.cfg.dataset.num_mel_bins,
            sampling_rate=self.cfg.dataset.sample_rate,
            hop_size=self.cfg.dataset.hop_size,
            win_size=self.cfg.dataset.win_size,
            fmin=self.cfg.dataset.f_min,
            fmax=self.cfg.dataset.f_max_loss
        )
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel)
        self.log("train/l1_loss", loss_mel, prog_bar=True)
        loss_mel = loss_mel * self.cfg.ml.loss.loss_scale_mel
        
        g_loss = loss_gen_s + loss_gen_u + loss_fm_s + loss_fm_u + loss_mel

        self.log("train/g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        if (batch_idx + 1) % self.cfg.ml.accumulate_grad_batches == 0:
            optimizer_g.step()
            optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        
        # train discriminator
        self.toggle_optimizer(optimizer_d)
        y_g_hats = self.model(x)
        detached_y_g_hats = [x.detach() for x in y_g_hats]
        y_du_hat_r, y_du_hat_g, _, _ = self.combd(ys, detached_y_g_hats)
        loss_disc_u, losses_disc_u_r, losses_disc_u_g = discriminator_loss(y_du_hat_r, y_du_hat_g)
        y_ds_hat_r, y_ds_hat_g, _, _ = self.sbd(y, detached_y_g_hats[-1])
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        d_loss = loss_disc_s + loss_disc_u
        self.log("train/d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        if (batch_idx + 1) % self.cfg.ml.accumulate_grad_batches == 0:
            optimizer_d.step()
            optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def on_train_epoch_end(self) -> None:
        if self.cfg.ml.scheduler.use:
            sh_g, sh_d = self.lr_schedulers()
            sh_g.step()
            sh_d.step()
        return None

    def on_validation_epoch_start(self) -> None:
        self._val_loss_list = []
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        y, x, y_mel, _ = batch
        assert x.shape[0] == 1, f"validation batch size should be 1"
        y_g_hat = self(x)
        y_g_hat_mel = mel_spectrogram_torch(
            y_g_hat.squeeze(1),
            n_fft=self.cfg.dataset.fft_size,
            num_mels=self.cfg.dataset.num_mel_bins,
            sampling_rate=self.cfg.dataset.sample_rate,
            hop_size=self.cfg.dataset.hop_size,
            win_size=self.cfg.dataset.win_size,
            fmin=self.cfg.dataset.f_min,
            fmax=self.cfg.dataset.f_max_loss
        )
        val_loss = F.l1_loss(y_mel, y_g_hat_mel)
        
        self.logger.experiment.add_audio(
            f'pred/{batch_idx}', y_g_hat.squeeze(), self.current_epoch, self.cfg.dataset.sample_rate)
        self.logger.experiment.add_audio(
            f'gt/{batch_idx}', y[0].squeeze(), self.current_epoch, self.cfg.dataset.sample_rate)
        self._val_loss_list.append(val_loss.item())
        
    def on_validation_epoch_end(self):
        assert len(self._val_loss_list) > 0
        val_loss = sum(self._val_loss_list) / (len(self._val_loss_list) + 1e-8)
        self.log("val_l1_loss", val_loss, prog_bar=False)
    
if __name__ == "__main__":
    from src.utils.conf import get_hydra_cnf
    cfg = get_hydra_cnf("src/conf", "config.yaml")
    AvocodoModule(cfg)
    
    