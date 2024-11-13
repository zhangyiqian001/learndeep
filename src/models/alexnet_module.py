from torch import nn
from torchvision.models import alexnet, AlexNet, AlexNet_Weights
from torchmetrics import Accuracy

from models.base_module import BaseModelModule

class AlexNetModule1x3(BaseModelModule):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.metrics = Accuracy("multiclass", num_classes=num_classes)
        self.model = AlexNet(num_classes, dropout)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.model.classifier.append(nn.Softmax(dim=1))
    # def transfer_batch_to_device(self, batch: dict, device: torch.device, dataloader_idx: int) -> Any:
    #     result = {}
    #     for key,value in batch.items():
    #         if isinstance(value, dict):
    #             result[key] = {k: v.to(device) for k,v in value}
    #         else:
    #             result[key] = value.to(device)
    #     in_channels = batch['inputs'].shape[1]
    #     return result

class AlexNetModule3x3(BaseModelModule):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.metrics = Accuracy("multiclass", num_classes=num_classes)
        self.model = AlexNet(num_classes, dropout)
        self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.model.classifier.append(nn.Softmax(dim=1))

class AlexNetModule_IMAGENET1KV1(BaseModelModule):
    def __init__(self, num_classes):
        super().__init__()
        weight = AlexNet_Weights.IMAGENET1K_V1
        self.loss = nn.CrossEntropyLoss()
        self.metrics = Accuracy("multiclass", num_classes=num_classes)
        self.model = alexnet(weight)
