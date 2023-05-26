from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torch.utils.data
import torchaudio
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wave(
    wave_file_path: Union[str, Path],
    sample_rate: int = -1,
    is_torch: bool = True,
    mono: bool = False,
) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
    """load wave

    Args:
        wave_file_path (str): file path
        sample_rate (int, optional): if -1 return original sample rate. Defaults to -1.
        is_torch (bool, optional): return torch.Tensor or np.ndarray. Defaults to True.
        mono (bool, optional):
            True: return [wave]
            False: return [channel, wave].
            Defaults to False.

    Returns:
        wave torch.Tensor or np.ndarray return
        sample_rate (int)
    """

    wave, sr = torchaudio.load(wave_file_path)
    if mono:
        wave = wave[0]
    if sample_rate > 0 and sample_rate != sr:
        wave = torchaudio.transforms.Resample(sr, sample_rate)(wave)
    else:
        sample_rate = sr
    if not is_torch:
        wave = wave.cpu().detach().numpy().copy()
    return wave, sample_rate


def save_wave(
    wave: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path],
    sample_rate: int = 16000,
) -> None:
    """save wave"""
    if not isinstance(wave, torch.Tensor):
        wave = torch.from_numpy(wave)

    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    torchaudio.save(
        filepath=str(output_path), src=wave.to(torch.float32), sample_rate=sample_rate
    )


def dynamic_range_compression_torch(
    x: torch.Tensor, C: int = 1, clip_val: float = 1e-5
) -> torch.Tensor:
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x: torch.Tensor, C: int = 1) -> torch.Tensor:
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(
    spec: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: int,
    fmax: int,
) -> torch.Tensor:
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int,
    center: bool = False,
) -> torch.Tensor:
    assert y.dim() == 2
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
