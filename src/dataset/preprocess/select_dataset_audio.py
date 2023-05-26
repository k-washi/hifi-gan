# VCのデータセットとして使用するデータを抽出 + 加工

import random
from pathlib import Path

import torch
import librosa
from librosa.util import normalize
from tqdm import tqdm
from typing import Union, List

from src.utils.audio import load_wave, save_wave
from src.utils.logger import get_logger

logger = get_logger(debug=True)

SEED = 3407
random.seed(SEED)

MAX_WAV_VALUE = 32768.0


def select_dataset_audio(
    dataset_dir_list: List[Union[str, Path]], output_dir: Union[str, Path], sample_rate:int, audio_time_sec_max:float, spk_time_sec_max:float
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for dataset_dir in dataset_dir_list:
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"{dataset_dir} is not found")
        spk_dirs = sorted(list(dataset_dir.glob("*")))
        for spk_dir in tqdm(spk_dirs):
            spk_out_dir = output_dir / spk_dir.name
            spk_out_dir.mkdir(exist_ok=True, parents=True)
            spk_wave_out_dir = spk_out_dir / "wav"
            spk_wave_out_dir.mkdir(exist_ok=True, parents=True)

            audio_list = sorted(list((spk_dir / "wav").glob("*")))
            random.shuffle(audio_list)
            total_time = 0.0
            for audio_file in audio_list:
                waveform = load_wave(
                    str(audio_file), sample_rate=sample_rate, is_torch=True, mono=True
                )[0]

                # 最大の音源長さをaudio_time_sec_maxに合わせる
                if waveform.shape[0] / sample_rate > audio_time_sec_max:
                    waveform = waveform[: int(audio_time_sec_max * sample_rate)]
                waveform = torch.Tensor(waveform).numpy()/ MAX_WAV_VALUE
                waveform = normalize(waveform) * 0.95
                waveform, _ = librosa.effects.trim(waveform)
                # 話者ごとの音源時間を加算
                waveform_time = waveform.shape[0] / sample_rate
                total_time += waveform_time
                if total_time > spk_time_sec_max:
                    total_time -= waveform_time
                    break

                save_wave(
                    waveform,
                    str(spk_wave_out_dir / audio_file.name),
                    sample_rate=int(sample_rate),
                )

            logger.info(f"{spk_dir.name} has {total_time:.2f} sec")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir_list",
        type=str,
        nargs="*",
        required=True,
        help="dataset directory list",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="output directory"
    )
    parser.add_argument(
        "-sr", "--sample_rate", type=int, default=22050, help="sample rate"
    )
    parser.add_argument(
        "--audio_time_sec_max", type=float, default=15, help="audio time max"
    )
    parser.add_argument(
        "--spk_time_sec_max", type=float, default=2700, help="audio time sum max by spk"
    )

    args = parser.parse_args()

    select_dataset_audio(
        args.dataset_dir_list,
        args.output_dir,
        args.sample_rate,
        args.audio_time_sec_max,
        args.spk_time_sec_max,
    )
