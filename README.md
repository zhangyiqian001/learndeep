# learndeep

#### 项目介绍
基于 PyTorch Lightning 框架

经典模型复现:
- GAN
- Transformer
- Bert
- PixelCNN
- Gated PixelCNN
- PixelCNN++
- VAE
- VQ-VAE
- VQ-VAE2


#### 训练脚本
```sh
python .\main.py fit -c .\config.yaml
```

#### 可视化脚本
```sh
tensorboard --logdir .\\logger\\
```