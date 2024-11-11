from utils.registry import registry
from .base_task import BaseTask
from .image_classification_task import ImageClassificationTask


def create_task(config):
    assert "TASK" in config.BASE, "Task name must be provided."
    task_name = config.BASE.TASK
    task = registry.get_task_class(task_name).setup_task(config=config)
    assert task is not None, "Task {} not properly registered.".format(task_name)
    return task