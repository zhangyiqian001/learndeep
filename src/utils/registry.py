"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


class Registry:
    mapping = {
        "datamodule_name_mapping": {},
        "task_name_mapping": {},
        "processor_name_mapping": {},
        "model_name_mapping": {},
    }

    @classmethod
    def register_datamodule(cls, name):
        r"""Register a datamodule to registry with key 'name'

        Args:
            name: Key with which the datamodule will be registered.

        Usage:

            from lavis.common.registry import registry
            from lavis.datamodules.base_datamodule_datamodule import BaseDatamodule
        """

        def wrap(datamodule_cls):
            from datasets import BaseDataModule

            assert issubclass(datamodule_cls, BaseDataModule), "All datamodules must inherit BaseDataModule class, found {}".format(datamodule_cls)
            if name in cls.mapping["datamodule_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["datamodule_name_mapping"][name]
                    )
                )
            cls.mapping["datamodule_name_mapping"][name] = datamodule_cls
            return datamodule_cls

        return wrap

    @classmethod
    def register_task(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from lavis.common.registry import registry
        """

        def wrap(task_cls):
            from tasks import BaseTask

            assert issubclass(task_cls, BaseTask), "All tasks must inherit BaseTask class, found {}".format(task_cls)
            if name in cls.mapping["task_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["task_name_mapping"][name]
                    )
                )
            cls.mapping["task_name_mapping"][name] = task_cls
            return task_cls

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from lavis.common.registry import registry
        """

        def wrap(model_cls):
            from models import BaseModelModule

            assert issubclass(model_cls, BaseModelModule), "All models must inherit BaseModelModule class, found {}".format(model_cls)
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def register_processor(cls, name):
        r"""Register a processor to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from lavis.common.registry import registry
        """

        def wrap(processor_cls):
            from processors import BaseProcessor

            assert issubclass(processor_cls, BaseProcessor), "All processors must inherit BaseProcessor class, found {}".format(processor_cls)
            if name in cls.mapping["processor_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["processor_name_mapping"][name]
                    )
                )
            cls.mapping["processor_name_mapping"][name] = processor_cls
            return processor_cls

        return wrap

    @classmethod
    def get_datamodule_class(cls, name):
        return cls.mapping["datamodule_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_tasks(cls):
        return sorted(cls.mapping["task_name_mapping"].keys())

    @classmethod
    def list_processors(cls):
        return sorted(cls.mapping["processor_name_mapping"].keys())

    @classmethod
    def list_datamodules(cls):
        return sorted(cls.mapping["datamodule_name_mapping"].keys())



registry = Registry()
