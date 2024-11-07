from pathlib import Path

from omegaconf import OmegaConf

from tasks import BaseTask
from utils.registry import registry
from utils.trainers import create_trainer
@registry.register_task("image_classification")
class ImageClassificationTask(BaseTask):
    def __init__(
            self,
            config,
            batch_size,
            epoch,
            loss,
            optim,
            metrics,
            resume,
            pretrained
    ):
        super().__init__(config)


    def build(self, ):
        self.build_model()
        create_trainer()
    @classmethod
    def setup_task(cls, config):
        train_config = config.TRAIN
        batch_size = train_config.get('BATCH_SIZE', '64')
        epoch = train_config.get('END_EPOCH', '3001')
        loss = train_config.get('LOSS', 'CrossEntropyLoss')
        optim = train_config.get('BATCH_SIZE', '64')
        metrics = train_config.get('BATCH_SIZE', '64')
        resume = train_config.get('RESUME', None)
        pretrained = train_config.get('PRETRAINED', None)

        return cls(
            config,
            batch_size,
            epoch,
            loss,
            optim,
            metrics,
            resume,
            pretrained
        )
