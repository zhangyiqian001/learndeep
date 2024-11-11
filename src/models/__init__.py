from .base_module import BaseModelModule
from .text_classification import *
from .image_classification import *

import sys

__all__ = [

]

def create_model(config):
    task = config.BASE.TASK
    model_type = config.MODEL.MODEL_TYPE + "Module"
    module = sys.modules[f"models.{task}"]
    Model = module.__getattribute__(model_type)
    return Model(config)
