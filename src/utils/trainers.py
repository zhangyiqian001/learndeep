import os
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy


def create_trainer(config):
    vis_loggers = create_logger(config)
    callbacks = create_callback(config)
    if len(config.DEVICE) > 1:
        ddp_strategy = "ddp"
    else:
        ddp_strategy = 'auto'
    trainer = L.Trainer(
        fast_dev_run=config.DEBUG,
        benchmark=True,
        max_epochs=config.TRAIN.END_EPOCH,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICE,
        strategy=ddp_strategy,
        default_root_dir=config.FOLDER_EXP,
        # 在步骤内多久记录一次, default: 50
        log_every_n_steps=config.LOGGER.LOG_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=vis_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=config.LOGGER.CHECK_VAL_EPOCH,
    )

    return trainer


def create_logger(config):
    vis_loggers = []

    if config.LOGGER.WANDB.PROJECT:
        wandb_logger = WandbLogger(
            project=config.LOGGER.WANDB.PROJECT,
            offline=config.LOGGER.WANDB.OFFLINE,
            id=config.LOGGER.WANDB.RESUME_ID,
            save_dir=config.FOLDER_EXP,
            version="",
            name=config.NAME,
            anonymous=False,
            log_model=False,
        )
        vis_loggers.append(wandb_logger)

    if config.LOGGER.TENSORBOARD:
        tb_logger = TensorBoardLogger(
            save_dir=config.FOLDER_EXP,
            sub_dir="tensorboard",
            version="",
            name=""
        )
        vis_loggers.append(tb_logger)
    return vis_loggers


def create_callback(config):
    return [
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
        # ProgressLogger(metric_monitor=metric_monitor),
        ModelCheckpoint(
            dirpath=os.path.join(config.FOLDER_EXP, "checkpoints"),
            filename="{epoch}",
            monitor="step",
            mode="max",
            every_n_epochs=config.LOGGER.SAVE_CHECKPOINT_EPOCH,
            save_top_k=-1,  # 根据monitor保存最好的几个
            save_last=False,
            save_on_train_epoch_end=True,
        ),
    ]
