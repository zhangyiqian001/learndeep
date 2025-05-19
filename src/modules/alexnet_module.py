from torch import nn

from modules.base_module import BaseClassificationModule


class AlexNetModuleBase(BaseClassificationModule):
    def __init__(
            self,
            model,
            loss,
            metrics
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.metrics = metrics


class AlexNetModule1x3(AlexNetModuleBase):
    def __init__(
            self,
            model,
            loss,
            metrics
    ):
        super().__init__(
            model,
            loss,
            metrics
        )
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)


class AlexNetModule3x3(AlexNetModuleBase):
    def __init__(
            self,
            model,
            loss,
            metrics
    ):
        super().__init__(
            model,
            loss,
            metrics
        )
        self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
