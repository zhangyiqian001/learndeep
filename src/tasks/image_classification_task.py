from omegaconf import OmegaConf
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

    def run(self, stage="train"):
        datamodule = self.build_datamodule()
        self.module_config['ARCH_CONFIG']['num_classes'] = datamodule.num_classes
        self.dynamic_config['metrics_args']['num_classes'] = datamodule.num_classes
        module = self.build_module(OmegaConf.to_container(self.module_config) | self.dynamic_config)
        module.model.classifier.append(nn.Softmax(dim=1))
        self.trainer = self.build_trainer()
        if stage == "train":
            self.trainer.fit(module, datamodule, ckpt_path=self.pretrained)
        elif stage == "test":
            self.trainer.test(module, datamodule, ckpt_path=self.pretrained)

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
