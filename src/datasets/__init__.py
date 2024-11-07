from .base_datamodule import BaseDataModule
from .datamodule import *
import sys


def create_dataset(config):
    # get dataset names form config
    origin = config.DATASET.ORIGIN
    module = sys.modules[f"datasets"]
    Dataset = module.__getattribute__(origin)
    dataset = Dataset(config=config)
    dataset.prepare_data()
    return dataset