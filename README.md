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

# model convert

pytorch lightning module から、pytorch module　に変更

```
python ./src/helper/plmodule_to_torch.py \
--checkpoint checkpoints/00002/checkpoint-epoch=0426-val_l1_loss=0.0823.ckpt \
--output ./data/avocodo_v1.pth
```

# tesnsorboard

```
tensorboard --logdir=./logs/ --host=0.0.0.0 --port=18282
```


# Ref

[Avocodo: Generative Adversarial Network for Artifact-Free Vocoder](https://github.com/ncsoft/avocodo/tree/main)

# モデル

|モデル|description|
|-|-|
|avocodo_v1.pth| 正規化音で学習したボコーダー|

```
gdown https://drive.google.com/u/1/uc?id=1GgLsmDRqVM9JzT7HcUYW8BR0dh5a1IPm -O avocodo_v1.pth
```