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

常用数据集:
- [huggingface_datamodule.py](src%2Fdatamodules%2Fhuggingface_datamodule.py)
- [modelscope_datamodule.py](src%2Fdatamodules%2Fmodelscope_datamodule.py)
- [torchtext_datamodule.py](src%2Fdatamodules%2Ftorchtext_datamodule.py)
- [torchvision_datamodule.py](src%2Fdatamodules%2Ftorchvision_datamodule.py)
- [vctk_datamodule.py](src%2Fdatamodules%2Fvctk_datamodule.py)
- [carvana_datamodule.py](src%2Fdatamodules%2Fcarvana_datamodule.py)



#### 训练脚本
```sh
python .\main.py fit -c .\config.yaml
```

#### 可视化脚本
```sh
tensorboard --logdir .\\logger\\
```