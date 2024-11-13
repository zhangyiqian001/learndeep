"""
保证任务，模型，数据一致
- 注册任务，模型，数据在字典中
- 验证参数是否正确
- 子类任务有任务特定的loss，metric，必要参数
"""
import os
from pathlib import Path

from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping, RichModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from omegaconf import OmegaConf

from utils.registry import registry

import lightning as L


class BaseTask:
    def __init__(self, config):
        self.task = config.BASE.TASK
        self.accelerator = config.BASE.ACCELERATOR
        self.device = config.BASE.DEVICE
        self.debug = config.BASE.DEBUG
        self.config_root = registry.get_path("config_root")

        self.processor_config = config.PROCESSOR
        self.datamodule_config = config.DATAMODULE
        self.module_config = config.MODEL

        self.epoch = config.TRAIN.EPOCH
        self.pretrained = config.TRAIN.PRETRAINED

    def build_datamodule(self):
        if self.processor_config.NAME:
            registry.get_processor_class(self.processor_config.NAME).setup()
        datamodule = registry.get_datamodule_class(self.datamodule_config.NAME).from_config(self.datamodule_config)
        datamodule.prepare_data()
        return datamodule

    def build_module(self, config):
        config = OmegaConf.create(config)
        return registry.get_model_class(config.NAME).from_config(config)

    def build_trainer(self):
        vis_loggers = []
        log_config = OmegaConf.load(os.path.join(self.config_root, "assets.yaml"))
        save_dir = Path(log_config.FOLDER_EXP) / self.task / (self.datamodule_config.NAME + '_' + self.module_config.NAME)
        self.pretrained = save_dir / self.pretrained
        if log_config.WANDB.PROJECT:
            wandb_logger = WandbLogger(
                project=log_config.WANDB.PROJECT,
                offline=log_config.WANDB.OFFLINE,
                id=log_config.WANDB.RESUME_ID,
                save_dir=str(save_dir),
                version="",
                name=self.datamodule_config.NAME + '_' + self.module_config.NAME,
                anonymous=False,
                log_model=False,
            )
            vis_loggers.append(wandb_logger)

        if log_config.TENSORBOARD:
            tb_logger = TensorBoardLogger(
                save_dir=str(save_dir),
                sub_dir="tensorboard",
                version="",
                name=""
            )
            vis_loggers.append(tb_logger)

        callbacks =  [
            RichProgressBar(
                theme=RichProgressBarTheme(
                    description="green_yellow",
                    progress_bar="green1",
                    progress_bar_finished="green1",
                    progress_bar_pulse="#6206E0",
                    batch_progress="green_yellow",
                    time="grey82",
                    processing_speed="grey82",
                    metrics="grey82",
                    metrics_text_delimiter="\n",
                    metrics_format=".3e",
                )),
            EarlyStopping(
                monitor='val_loss',
                mode="min"
            ),
            ModelCheckpoint(
                dirpath=str(save_dir / "checkpoints"),
                filename="{epoch}",
                monitor="val_loss",
                mode="min",
                every_n_epochs=log_config.SAVE_CHECKPOINT_EPOCH,
                save_top_k=5,
                save_last=False,
                save_on_train_epoch_end=True,
            ),
            RichModelSummary()
        ]

        if len(self.device) > 1:
            ddp_strategy = "ddp"
        else:
            ddp_strategy = 'auto'

        trainer = L.Trainer(
            fast_dev_run=self.debug,
            max_epochs=self.epoch,
            accelerator=self.accelerator,
            devices=self.device,
            strategy=ddp_strategy,
            # 在步骤内多久记录一次, default: 50
            log_every_n_steps=log_config.LOG_EVERY_STEPS,
            deterministic=False,
            detect_anomaly=False,
            enable_progress_bar=True,
            logger=vis_loggers,
            callbacks=callbacks,
            check_val_every_n_epoch=log_config.CHECK_VAL_EPOCH,
        )
        return trainer

    def run(self, stage="train"):
        pass

    @classmethod
    def setup_task(cls, config):
        pass
