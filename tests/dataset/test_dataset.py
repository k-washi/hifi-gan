import math

import torch

from src.dataset.dataset import VoiceDataset

TEST_SRC = "tests/__example/test.wav"


def test_voice_dataset():
    hop_size = 160
    win_size = 512
    sr = 16000
    for waveform_length in [16000, 21456, 21556, 25600]:
        dataset = VoiceDataset(
            [TEST_SRC],
            hop_size=hop_size,
            win_size=win_size,
            sample_rate=sr,
            waveform_length=waveform_length,
        )

        waveform, mel, audio_path = next(iter(dataset))

        assert len(dataset) == 1, f"{len(dataset)}!= 1"
        assert TEST_SRC == audio_path, f"{audio_path}!= {TEST_SRC}"
        assert (
            waveform.size()[0] == waveform_length
        ), f"{waveform.size()}!= {waveform_length}"
        assert mel.size()[1] == math.floor(
            waveform.size()[0] / hop_size
        ), f"{mel.size()}!= {waveform.size()[0] / hop_size}"

    dataset = VoiceDataset([TEST_SRC], is_audio_path_only=True)
    waveform, mel, audio_path = next(iter(dataset))
    assert (
        waveform.size() == mel.size() == torch.Tensor([0]).size()
    ), f"{waveform.size()}!= {mel.size()}!= {torch.Tensor([0])}"
    assert audio_path == TEST_SRC, f"{audio_path}!= {TEST_SRC}"
