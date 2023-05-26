from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from audiomentations import Compose, TimeStretch
from librosa.util import normalize
from torch.utils.data import Dataset

from src.utils.audio import load_wave, mel_spectrogram_torch

MAX_WAV_VALUE = 32768


class VoiceDataset(Dataset):
    def __init__(
        self,
        audio_filelist: List[Union[str, Path]],
        waveform_length: int = 25600,
        is_aug: bool = False,
        time_stretch_params: Tuple[float, float, float] = (0.9, 1.1, 0.5),
        num_mel_bins: int = 80,
        hop_size: int = 160,
        win_size: int = 512,
        fft_size: int = 512,
        f_min: int = 0,
        f_max: int = 8000,
        f_max_loss: int = 8000,
        sample_rate: int = 16000,
        is_audio_path_only: bool = False,
    ) -> None:
        super().__init__()
        self._audio_filelist = audio_filelist
        self._waveform_length = waveform_length
        self._is_aug = is_aug

        self._sample_rate = sample_rate
        self._num_mel_bins = num_mel_bins
        self._hop_size = hop_size
        self._win_size = win_size
        self._fft_size = fft_size
        self._f_min = f_min
        self._f_max = f_max
        self._f_max_loss = f_max_loss

        self._is_audio_path_only = is_audio_path_only

        if is_aug:
            self.waveform_aug = Compose(
                [
                    TimeStretch(
                        min_rate=time_stretch_params[0],
                        max_rate=time_stretch_params[1],
                        p=time_stretch_params[2],
                    )
                ]
            )

    def __len__(self) -> int:
        return len(self._audio_filelist)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        audio_file = self._audio_filelist[index]
        if self._is_audio_path_only:
            return (
                torch.as_tensor([0]),
                torch.as_tensor([0]),
                torch.as_tensor([0]),
                str(audio_file),
            )
        waveform = load_wave(
            audio_file, sample_rate=self._sample_rate, is_torch=False, mono=True
        )[0]
        waveform = waveform / MAX_WAV_VALUE
        waveform = normalize(waveform) * 0.95

        if self._is_aug:
            waveform = self.waveform_aug(waveform, sample_rate=self._sample_rate)

        if waveform.shape[0] <= self._waveform_length:
            shortage = self._waveform_length - waveform.shape[0]
            waveform = np.pad(waveform, (0, shortage), "constant", constant_values=0)
        else:
            start_frame = np.int64(
                torch.rand(1).item() * (waveform.shape[0] - self._waveform_length)
            )
            waveform = waveform[start_frame : start_frame + self._waveform_length]
        # waveform = np.stack([waveform], axis=0)
        waveform = torch.FloatTensor(waveform)
        mel_spectrogram = mel_spectrogram_torch(
            waveform.unsqueeze(0),
            n_fft=self._fft_size,
            num_mels=self._num_mel_bins,
            sampling_rate=self._sample_rate,
            hop_size=self._hop_size,
            win_size=self._win_size,
            fmin=self._f_min,
            fmax=self._f_max,
        )

        mel_spectrogram = mel_spectrogram.squeeze(0)

        if self._f_max == self._f_max_loss:
            mel_loss = mel_spectrogram
        else:
            mel_loss = mel_spectrogram_torch(
                waveform.unsqueeze(0),
                n_fft=self._fft_size,
                num_mels=self._num_mel_bins,
                sampling_rate=self._sample_rate,
                hop_size=self._hop_size,
                win_size=self._win_size,
                fmin=self._f_min,
                fmax=self._f_max_loss,
            )
            mel_loss = mel_loss.squeeze(0)

        return waveform, mel_spectrogram, mel_loss, str(audio_file)
