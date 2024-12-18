from typing import Union, Any, Tuple, Dict

import matplotlib.pyplot as plt
from PIL.Image import Image, fromarray
from torchmetrics.text import BLEUScore
from torchtext.datasets import DATASETS
from torchvision.datasets import CIFAR10
from torchvision import transforms

# class A:
#     def __init__(self, args1, args2, a, b):
#         print(args1)
#         print(args2)
#         print(a)
#         print(b)
#
# def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
#     """Instantiates a class with the given args and init.
#
#     Args:
#         args: Positional arguments required for instantiation.
#         init: Dict of the form {"class_path":...,"init_args":...}.
#
#     Returns:
#         The instantiated class object.
#
#     """
#     kwargs = init.get("init_args", {})
#     class_module, class_name = init["class_path"].rsplit(".", 1)
#     module = __import__(class_module, fromlist=[class_name])
#     args_class = getattr(module, class_name)
#     return args_class(*args, **kwargs)
#
# def fun(a, b):
#     pass
if __name__ == '__main__':
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.CenterCrop(148),
                                           transforms.Resize(64),
                                           transforms.ToTensor(),])

    train_iter = CIFAR10("data", train=True, download=True, transform=train_transforms)
    # a = train_transforms(fromarray(train_iter.data[0]))
    a = train_iter.data[0]
    print(a.shape)
    # print(train_iter.data[0])
    # plt.imshow(a.permute(1, 2, 0).detach().numpy())
    plt.imshow(a)
    plt.show()
    # data = DATASETS['AG_NEWS']("data", split='train')
    # data = DATASETS['SQuAD2']("data", split='train')
    # print(data)
    # a = list(iter(data))
    # print(a[0])
    # print(a[269])
    # print(a[258])
    # print(a[161])
    # print(a[1])
    # for i in list(data):
    #     print(i)
    # preds = ['the cat is on the mat',]
    # target = [['a cat is on the mat'], ['a cat is on the mat']]
    # # target = [['there is a cat on the mat']]
    # metrix = BLEUScore(n_gram=1)
    # print(metrix(preds, target))
    # k = {
    #     "a": 1,
    #     "b": 3,
    #     "c": 2,
    # }
    # fun(**k)
    # args = [1,2]
    # init = {
    #     "class_path": "test.A",
    #     "init_args": {
    #         "a": 3,
    #         "b": 4,
    #     }
    # }
    # print(instantiate_class(args, init))