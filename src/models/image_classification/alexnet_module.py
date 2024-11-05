from torch import nn
from torchvision.models import AlexNet

from models.base_module import BaseModelModule


class AlexNetModule(BaseModelModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        args = {key.lower(): value for key, value in config.MODEL.ARCH_CONFIG.items()}
        if config.DATASET.NAME == "MNIST":
            self.model = AlexNet(**args)
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        else:
            self.model = AlexNet(**args)

