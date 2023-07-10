# 環境構築

```
docker-compose up -d
```

## Docker内

```
poetry install
```

## vscode extentions install

```
./.devcontainer/vscode_extentions_install_batch.sh 
```

# Preprocess

```
poetry run python ./src/dataset/preprocess/select_dataset_audio.py \
    --dataset_dir_list /data/jvs_vc/ /data/lecture_vc/ /data/common_voice/ /data/vc_dataset/ \
    --output_dir /data/voc16k \
    --sample_rate 16000 \
    --audio_time_sec_max 15 \
    --spk_time_sec_max 1800
```

# train

```
python ./pipeline/train/exp0001.py
```

# tesnsorboard

```
tensorboard --logdir=./logs/ --host=0.0.0.0 --port=18282
```


# Ref

[Avocodo: Generative Adversarial Network for Artifact-Free Vocoder](https://github.com/ncsoft/avocodo/tree/main)