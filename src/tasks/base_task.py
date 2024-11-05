"""
保证任务，模型，数据一致
- 注册任务，模型，数据在字典中
- 验证参数是否正确
- 子类任务有任务特定的loss，metric，必要参数
"""
class BaseTask:
    def __init__(self, config):
        self.config = config
        self.task = config.TASK

    def valid_args(self):
        if self.task == "image_classification":
            if self.config.DATASET.NAME in []:
                pass
            if self.config.MODEL.NAME in []:
                pass
    def build_dataset(self):
        pass


    def build_model(self):
        pass