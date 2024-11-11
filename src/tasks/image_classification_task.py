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
        self.model = None
        self.datamodule = None
        self.trainer = None


    def run(self):
        config = self.build_config()
        config = config | self.dynamic_config
        self.datamodule = self.build_datamodule(config)
        config['ARCH_CONFIG']['num_classes'] = self.datamodule.num_classes
        config['metrics_args']['num_classes'] = self.datamodule.num_classes
        self.model = self.build_model(config)
        self.trainer = self.build_trainer()
        self.trainer.fit(self.model, self.datamodule)

    @classmethod
    def setup_task(cls, config):
        train_config = config.TRAIN
        loss = train_config.LOSS.TYPE
        loss_args = train_config.LOSS.get('ARGS', {})
        optim = train_config.OPTIM.TYPE
        optim_args = train_config.OPTIM.get('ARGS', {})
        metrics = train_config.METRICS.TYPE
        metrics_args = train_config.METRICS.get('ARGS', {})
        resume = train_config.get('RESUME', None)
        pretrained = train_config.get('PRETRAINED', None)
        dynamic = {
            "loss": loss,
            "optim": optim,
            "metrics": metrics,
            "loss_args": loss_args,
            "optim_args": optim_args,
            "metrics_args": metrics_args,
            "pretrained": pretrained,
            "resume": resume,
        }
        return cls(config, dynamic)
