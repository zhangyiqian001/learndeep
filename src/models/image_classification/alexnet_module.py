from torchvision.models import AlexNet

from models.base_module import BaseModelModule


class AlexNetModule(BaseModelModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AlexNet(config.ARCH_CONFIG)
