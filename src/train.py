import json
from pathlib import Path

import torch
import typer
from omegaconf import OmegaConf

from tasks import create_task
from utils.registry import registry

import __init__

# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
app = typer.Typer()

def logger_info(config, key):
    OmegaConf.to_container(config)
    data = json.dumps(OmegaConf.to_container(config)[key], indent=4, ensure_ascii=False, sort_keys=False,
                      separators=(',', ':'))
    return data


# python .\train.py .\src\config\text_classification\transformer.yaml
@app.command()
def main(config_path):
    # torch.set_float32_matmul_precision('medium' | 'high')
    torch.set_float32_matmul_precision('medium')

    config = OmegaConf.load(config_path)
    print(registry.mapping)
    # 1、create logger
    # logger, _ = create_logger(config, phase="train")

    # logger.info(logger_info(config, "LOGGER"))

    # 3、set seed
    # L.seed_everything(config.BASE.SEED_VALUE)

    # 4、gpu setting
    # if config.BASE.ACCELERATOR == "gpu":
    #     os.environ["PYTHONWARNINGS"] = "ignore"
    #     os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #     os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(x) for x in config.BASE.DEVICE)

    # create task
    # logger.info(logger_info(config, "BASE"))
    # logger.info(registry.mapping)
    task = create_task(config)
    task.run()
    # checkpoint
    # checkpoint_folder = trainer.checkpoint_callback.dirpath
    # logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    # logger.info(
    #     f"The outputs of this experiment are stored in {config.FOLDER_EXP}")
    #
    # # end
    # logger.info("Training ends!")



if __name__ == '__main__':
    app()
    # main("config/text_classification/transformer.yaml")
