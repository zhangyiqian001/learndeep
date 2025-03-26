# learndeep

## 项目介绍
基于 PyTorch Lightning 框架

经典模型复现:
- Unet
- GAN
- PixelCNN
- Gated PixelCNN
- PixelCNN++
- Transformer
- Bert
- SwitchTransformer
- VAE
- VQ-VAE
- VQ-VAE2
- Diffusion
- ViT
- VQ-GAN

常用数据集:
- [huggingface_datamodule.py](src%2Fdatamodules%2Fhuggingface_datamodule.py)
  - [wmt/wmt14](https://huggingface.co/datasets/wmt/wmt14)
  - [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech)
  - [CSTR-Edinburgh/vctk](https://huggingface.co/datasets/CSTR-Edinburgh/vctk)
- [modelscope_datamodule.py](src%2Fdatamodules%2Fmodelscope_datamodule.py)
- [torchtext_datamodule.py](src%2Fdatamodules%2Ftorchtext_datamodule.py)
- [torchvision_datamodule.py](src%2Fdatamodules%2Ftorchvision_datamodule.py)
- [carvana_datamodule.py](src%2Fdatamodules%2Fcarvana_datamodule.py)
- [Dogs vs. Cats DataSet](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data)

### 安装
1. 创建 conda environment
    ```shell
    conda create -n learndeep python=3.10
    conda activate learndeep
   git clone --recursive https://github.com/zhangyiqian001/learndeep.git
    ```
2. 安装依赖
    
   [了解更多poetry](https://blog.csdn.net/weixin_43457608/article/details/142361076?spm=1001.2014.3001.5501)
    
   ```shell
   pip install poetry # 安装poetry
   poetry install # 第一次安装依赖
   poetry update # 更新依赖
    ```
### 自定义数据集
命名规范：数据集_datamodule.py

示例：[catsanddogs_datamodule.py](src%2Fdatamodules%2Fcatsanddogs_datamodule.py)
```python
# torch 定义
class CatsDogsDataset(Dataset):
    def __init__(self, ):
        pass
# lightning 定义
class CatsAndDogsDataModule(BaseDataModule):
    def __init__(self, ):
        pass
```

### 自定义数据集
命名规范：模型名_module.py

示例：[vit_module.py](src%2Fmodules%2Fvit_module.py)
```python
# torch 定义
class ViTModel(nn.Module):
    def __init__(self, ):
        pass
# lightning 定义
class ViTModule(BaseClassificationModule):
    def __init__(self, ):
        pass
```


### 自定义训练文件
示例：[vit_catanddog_config.yaml](src%2Fexample%2Fimage%2Fvit_catanddog_config.yaml)
```yaml
model:
  class_path: modules.vit_module.ViTModule
  init_args:
    model:
      class_path: modules.vit_module.ViTModel
      init_args:
        image_size: 224
        patch_size: 32
        num_classes: &num_classes 2
        dim: 128
        depth: 12
        heads: 8
        mlp_dim: 512
        pool: 'cls'
        channels: 3
        dim_head: 64
        dropout: 0.
        emb_dropout: 0.
    loss:
      class_path: torch.nn.CrossEntropyLoss
    metrics:
      class_path: torchmetrics.Accuracy
      init_args:
        task: multiclass
        num_classes: *num_classes


data:
  class_path: datamodules.CatsAndDogsDataModule
  init_args:
    kwargs:
      root: data/dogs-vs-cats/
      batch_size: 64
      num_workers: 4

```
#### 配置文件详解
[config.yaml](src%2Fexample%2Fconfig.yaml)

#### 训练脚本
```sh
python .\main.py fit -c .\config.yaml
```

#### 可视化脚本
```sh
tensorboard --logdir .\\logger\\
```
#### 拉取文档
```shell
git submodule update --init --recursive
```