import hydra
import torch
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.modules.vocoder_data_module import VocoderDataModule
from src.modules.avocode_module import AvocodoModule

from src.utils.logger import get_logger
logger = get_logger(debug=True)

##########
# PARAMS #
##########

EXP_ID = "00005"
LOG_SAVE_DIR = f"logs/{EXP_ID}"
MODEL_SAVE_DIR = f"checkpoints/{EXP_ID}"

FAST_DEV_RUN = False # 確認用の実行を行うか

# TRAIN PARAMS
NUM_EPOCHS = 1000
BATCH_SIZE = 16
AUDIO_WIN_SIZE = 1024
LEARNING_RATE = 1e-4

LOG_NAME = f"avocodo_vocoder_b{int(BATCH_SIZE)}_e{int(NUM_EPOCHS)}"

logger.info(f"LOG_NAME: {LOG_NAME}")

# ----------------------------
# seed
SEED = 3407
seed_everything(SEED, workers=True)

config_path = Path(__file__).resolve().parent.parent.parent / "src/conf"
@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def train(cfg: DictConfig):
    
    cfg.ml.num_epochs = NUM_EPOCHS
    cfg.ml.batch_size = BATCH_SIZE
    cfg.ml.optim.learning_rate = LEARNING_RATE
    cfg.dataset.win_size = AUDIO_WIN_SIZE
    cfg.dataset.fft_size = AUDIO_WIN_SIZE
    
    logger.info(cfg)
    
    ################################
    # データセットとモデルの設定
    ################################
    
    dataset = VocoderDataModule(cfg)
    model = AvocodoModule(cfg)
    
    ################################
    # コールバックなど訓練に必要な設定
    ################################
    # ロガー
    tflogger = TensorBoardLogger(save_dir=LOG_SAVE_DIR, name=LOG_NAME, version=EXP_ID)
    tflogger.log_hyperparams(cfg)
    # モデル保存
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_SAVE_DIR,
        filename="checkpoint-{epoch:04d}-{val_l1_loss:.4f}",
        save_top_k=cfg.ml.model_save.top_k,
        monitor=cfg.ml.model_save.monitor,
        mode=cfg.ml.model_save.mode
    )
    
    
    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
        
        
     ################################
    # 訓練
    ################################
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
            precision=cfg.ml.mix_precision,
            accelerator=device,
            devices=cfg.ml.gpu_devices,
            max_epochs=cfg.ml.num_epochs,
            profiler=cfg.ml.profiler,
            fast_dev_run=FAST_DEV_RUN,
            logger=tflogger,
            callbacks=callback_list
        )
    logger.debug("START TRAIN")
    trainer.fit(model, dataset)
    
    # model チェックポイントの保存
    best_model_path = f"{MODEL_SAVE_DIR}/best_model.ckpt"
    if len(checkpoint_callback.best_model_path):
        logger.info(f"BEST MODEL: {checkpoint_callback.best_model_path}")
        logger.info(f"BEST SCORE: {checkpoint_callback.best_model_score}")
        _ckpt = torch.load(checkpoint_callback.best_model_path)
        model.load_state_dict(_ckpt["state_dict"])
        torch.save(model.model.state_dict(), best_model_path)
        logger.info(f"To BEST MODEL: {best_model_path}")
        # FOR LOAD
        # _ckpt = torch.load(f"{cfg.ml.model_save.save_dir}/{cfg.ml.log_name}/{cfg.ml.version}/best_model.ckpt")
        # model.model.load_state_dict(_ckpt)
    else:
        print("best model is not exist.")
        
if __name__ == "__main__":
    train()