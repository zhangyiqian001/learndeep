import json
import os
from pathlib import Path

import lightning as L
import torch
import typer
from omegaconf import OmegaConf

from tasks import create_task
from utils.logger import create_logger
from models import create_model
from datasets import create_dataset
from utils.trainers import create_trainer

# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
app = typer.Typer()

def logger_info(config, key):
    OmegaConf.to_container(config)
    data = json.dumps(OmegaConf.to_container(config)[key], indent=4, ensure_ascii=False, sort_keys=False,
                      separators=(',', ':'))
    return data

def setup_config(config):
    config.BASE.ROOT = str(Path(".").absolute())
    config.BASE.CONFIG_ROOT = str(Path(".").absolute() / "configs" / config.BASE.TASK)

# python .\train.py .\config\text_classification\transformer.yaml
@app.command()
def main(config_path):
    # torch.set_float32_matmul_precision('medium' | 'high')
    torch.set_float32_matmul_precision('medium')

    config = OmegaConf.load(config_path)
    setup_config(config)

    # 1、create logger
    logger, _ = create_logger(config, phase="train")

    logger.info(logger_info(config, "LOGGER"))

    # 2、resume
    if config.TRAIN.RESUME:
        resume = config.TRAIN.RESUME
    # 3、set seed
    L.seed_everything(config.BASE.SEED_VALUE)

    # 4、gpu setting
    if config.BASE.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(x) for x in config.BASE.DEVICE)

    # create task
    logger.info(logger_info(config, "BASE"))
    task = create_task(config)
    task.run()
    # checkpoint
    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(
        f"The outputs of this experiment are stored in {config.FOLDER_EXP}")

    # end
    logger.info("Training ends!")



if __name__ == '__main__':
    app()
    # main("config/text_classification/transformer.yaml")
