from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from pathlib import Path

from src.dataset.dataset import VoiceDataset
from src.utils.logger import get_logger
logger = get_logger(debug=True)


class VocoderDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
        self._train_datset_dir = cfg.dataset.train_dataset_dir
        self._val_datset_dir = cfg.dataset.val_dataset_dir
        
        self._train_audio_files = sorted(list(Path(self._train_datset_dir).rglob('*.wav')))
        assert len(self._train_audio_files) > 0, f"No training audio files found in {self._train_datset_dir}"
        logger.info(f"Number of training audio files: {len(self._train_audio_files)}")
        
        self._val_audio_files = sorted(list(Path(self._val_datset_dir).rglob('*.wav')))
        assert len(self._val_audio_files) > 0
        logger.info(f"Number of validation audio files: {len(self._val_audio_files)}")
        
        self._waveform_length = cfg.dataset.train.waveform_length
        self._validate_waveform_length = cfg.dataset.train.validate_waveform_length
        self._time_stretch_params = cfg.dataset.aug.time_stretch
        self._num_bel_bins = cfg.dataset.num_mel_bins
        self._hop_size = cfg.dataset.hop_size
        self._n_fft = cfg.dataset.fft_size
        self._win_size = cfg.dataset.win_size
        self._sample_rate = cfg.dataset.sample_rate
        self._f_min = cfg.dataset.f_min
        self._f_max = cfg.dataset.f_max
        self._f_max_loss = cfg.dataset.f_max_loss
        self._volume_mul_params = cfg.dataset.aug.volume_mul_params
        self._volume_aug_rate = cfg.dataset.aug.volume_aug_rate

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = VoiceDataset(
            self._train_audio_files,
            waveform_length=self._waveform_length,
            is_aug=True,
            time_stretch_params=self._time_stretch_params,
            num_mel_bins=self._num_bel_bins,
            hop_size=self._hop_size,
            fft_size=self._n_fft,
            win_size=self._win_size,
            sample_rate=self._sample_rate,
            f_min=self._f_min,
            f_max=self._f_max,
            f_max_loss=self._f_max_loss,
            volume_mul_params=self._volume_mul_params,
            volume_aug_rate=self._volume_aug_rate
        )
        
        return DataLoader(
            dataset,
            batch_size=self.cfg.ml.batch_size,
            num_workers=self.cfg.ml.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
    
    @rank_zero_only
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = VoiceDataset(
            self._val_audio_files,
            waveform_length=self._validate_waveform_length,
            is_aug=False,
            num_mel_bins=self._num_bel_bins,
            hop_size=self._hop_size,
            fft_size=self._n_fft,
            win_size=self._win_size,
            sample_rate=self._sample_rate,
            f_min=self._f_min,
            f_max=self._f_max,
            f_max_loss=self._f_max_loss
        )
        
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
        