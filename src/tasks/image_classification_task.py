from torch import nn

from tasks import BaseTask
from utils.registry import registry


@registry.register_task("image_classification")
class ImageClassificationTask(BaseTask):
    def __init__(
            self,
            config,
            dynamic_config
    ):
        super().__init__(config)
        self.dynamic_config = dynamic_config

        self.processor = None
        self.module = None
        self.datamodule = None
        self.trainer = None


    def run(self, stage="train"):
        config = self.build_config()
        config = config | self.dynamic_config
        self.datamodule = self.build_datamodule(config)
        config['ARCH_CONFIG']['num_classes'] = self.datamodule.num_classes
        config['metrics_args']['num_classes'] = self.datamodule.num_classes
        self.module = self.build_module(config)
        self.module.model.classifier.append(nn.Softmax(dim=1))
        self.trainer = self.build_trainer()
        if stage == "train":
            self.trainer.fit(self.module, self.datamodule, ckpt_path=self.pretrained)
        elif stage == "test":
            self.trainer.test(self.module, self.datamodule, ckpt_path=self.pretrained)

    @classmethod
    def setup_task(cls, config):
        train_config = config.TRAIN
        loss = train_config.LOSS.TYPE
        loss_args = train_config.LOSS.get('ARGS', {})
        optim = train_config.OPTIM.TYPE
        optim_args = train_config.OPTIM.get('ARGS', {})
        metrics = train_config.METRICS.TYPE
        metrics_args = train_config.METRICS.get('ARGS', {})
        dynamic = {
            "loss": loss,
            "optim": optim,
            "metrics": metrics,
            "loss_args": loss_args,
            "optim_args": optim_args,
            "metrics_args": metrics_args,
        }
        return cls(config, dynamic)
