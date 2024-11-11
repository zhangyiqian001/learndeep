from torchvision import transforms

from processors import BaseProcessor
from utils.registry import registry


@registry.register_processor("base_image_processor")
class BaseImageProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, item):
        return self.transform(item)


@registry.register_processor("image_classification_processor")
class ImageClassificationProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, item):
        return self.transform(item)