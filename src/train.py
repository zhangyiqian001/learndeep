import json
import os

import lightning as L
import torch
import typer
from omegaconf import OmegaConf

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


# python .\train.py .\config\transformer.yaml
@app.command()
def main(config_path):
    config = OmegaConf.load(config_path)

    # 1、create logger
    logger, _ = create_logger(config, phase="train")

    logger.info(logger_info(config, "LOGGER"))

    # 2、resume
    if config.TRAIN.RESUME:
        resume = config.TRAIN.RESUME
    # 3、set seed
    L.seed_everything(config.SEED_VALUE)

    # 4、gpu setting
    if config.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(x) for x in config.DEVICE)

    # create dataset
    logger.info(logger_info(config, "DATASET"))
    dataset = create_dataset(config)
    # create model
    logger.info(logger_info(config, "MODEL"))
    model = create_model(config)
    # trainer
    logger.info(logger_info(config, "TRAIN"))
    trainer = create_trainer(config)
    # load pretrained
    if config.TRAIN.PRETRAINED:
        logger.info("Loading pretrain mode from {}".format(
            config.TRAIN.PRETRAINED))
        logger.info("Attention! VAE will be recovered")
        state_dict = torch.load(config.TRAIN.PRETRAINED,
                                map_location="cpu")["state_dict"]
        # remove mismatched and unused params
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # if k not in ["denoiser.sequence_pos_encoding.pe"]:
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    # fitting
    if config.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=dataset,
                    ckpt_path=config.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=dataset)

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
