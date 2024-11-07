from torchvision import transforms

from processors import BaseProcessor

class BaseImageProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([])

    def __call__(self, item):
        return self.transform(item)