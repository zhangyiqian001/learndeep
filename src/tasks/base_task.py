"""
保证任务，模型，数据一致
- 注册任务，模型，数据在字典中
- 验证参数是否正确
- 子类任务有任务特定的loss，metric，必要参数
"""
from pathlib import Path

from omegaconf import OmegaConf

from utils.registry import registry

import lightning as L

class BaseTask:
    def __init__(self, config):
        self.task = config.BASE.TASK
        self.config_root = config.BASE.CONFIG_ROOT
        self.processor_name = config.PROCESSOR.NAME
        self.processor_config = config.PROCESSOR.CONFIG
        self.datamodule_name = config.DATAMODULE.NAME
        self.datamodule_config = config.DATAMODULE.CONFIG
        self.model_name = config.MODEL.NAME
        self.model_config = config.MODEL.CONFIG

    def build_processor(self):
        config_file = Path(self.config_root) / self.task / "tokenizer" / self.processor_name / self.processor_config
        config = OmegaConf.load(config_file)
        processor = registry.get_processor_class(self.datamodule_name).from_config(config)
        return processor

    def build_datamodule(self):
        config_file = Path(self.config_root) / self.task / "datasets" / self.datamodule_name / self.datamodule_config
        config = OmegaConf.load(config_file)
        datamodule = registry.get_datamodule_class(self.datamodule_name).from_config(config)
        return datamodule

    def build_model(self):
        config_file = Path(self.config_root) / self.task / "models" / self.model_name / self.model_config
        config = OmegaConf.load(config_file)
        model = registry.get_model_class(self.datamodule_name).from_config(config)
        return model

    def build_trainer(self):
        if len(config.BASE.DEVICE) > 1:
            ddp_strategy = "ddp"
        else:
            ddp_strategy = 'auto'
        trainer = L.Trainer(
            fast_dev_run=config.BASE.DEBUG,
            benchmark=True,
            max_epochs=config.TRAIN.END_EPOCH,
            accelerator=config.BASE.ACCELERATOR,
            devices=config.BASE.DEVICE,
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
    def build(self):
        pass

    @classmethod
    def setup_task(cls, config):
        pass