# learndeep

#### 项目介绍
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

#### 安装
1. 创建 conda environment
    ```shell
    conda create -n learndeep python=3.10
    conda activate learndeep
    ```
2. 安装依赖
    
   [了解更多poetry](https://blog.csdn.net/weixin_43457608/article/details/142361076?spm=1001.2014.3001.5501)
    
   ```shell
   pip install poetry # 安装poetry
   poetry install # 第一次安装依赖
   poetry update # 更新依赖
    ```
#### 训练脚本
```sh
python .\main.py fit -c .\config.yaml
```

#### 可视化脚本
```sh
tensorboard --logdir .\\logger\\
```