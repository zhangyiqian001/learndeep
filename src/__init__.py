import os

from omegaconf import OmegaConf

from models import *
from tasks import *
from tokenizers import *
from datasets import *

root_dir = os.path.dirname(os.path.abspath(__file__))
registry.register_path("root", root_dir)
root_dir = os.path.join(root_dir, "configs")
registry.register_path("config_root", root_dir)
cache_root = os.path.join(root_dir, "cache")
registry.register_path("cache_root", cache_root)
