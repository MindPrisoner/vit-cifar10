# ViT CIFAR10

这是一个在 CIFAR10 数据集上实现 Vision Transformer 的入门项目，用来对比 ViT 和传统 CNN 在小型图像分类任务上的训练表现。

项目重点关注：

- Patch Embedding 的实现
- Transformer Encoder 的使用
- 学习率调度对训练稳定性的影响
- ViT 在 CIFAR10 上的训练难度与最终效果

## 项目目标

- 使用 ViT 完成 CIFAR10 十分类任务
- 理解图像切块、token 化和位置编码
- 观察 Transformer 结构在图像任务上的训练特点
- 记录不同超参数下的训练表现

## 目录结构

```text
vit_cifar10/
├── configs/
│   └── config.py          # 训练超参数
├── datasets/
│   └── cifar10_dataset.py # CIFAR10 加载与增强
├── engine/
│   ├── trainer.py        # 训练逻辑
│   └── evaluator.py      # 评估逻辑
├── models/
│   └── vit.py            # ViT 模型实现
├── utils/
│   ├── logger.py         # TensorBoard 日志
│   └── seed.py           # 随机种子固定
├── train.py              # 训练入口
├── test.py               # 测试脚本预留
├── requirements.txt      # 依赖列表
└── README.md
```

## 数据集

项目使用 `torchvision.datasets.CIFAR10`。

数据增强和归一化策略：

- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`
- `ToTensor()`
- CIFAR10 标准均值方差归一化

测试集只做 `ToTensor()` 和归一化，不做增强。

> 注意：`download=False`，表示你需要提前把 CIFAR10 数据集放到 `./data` 下。

## 模型说明

`models/vit.py` 中实现了一个轻量版 ViT，核心结构如下：

- `PatchEmbedding`
  - 使用卷积把图像切成 patch
  - 将 patch 映射为 embedding token
- `cls_token`
  - 作为分类 token
- `pos_embed`
  - 可学习位置编码
- `nn.TransformerEncoder`
  - 6 层 Transformer 编码器
- `mlp_head`
  - 使用 `cls_token` 的输出做分类

该实现的输入是 32x32 的 CIFAR10 图像，默认 patch size 为 4，因此每张图会被切成 64 个 patch。

## 训练流程

安装依赖后运行：

```bash
pip install -r requirements.txt
python train.py
```

训练入口会：

1. 固定随机种子
2. 加载 CIFAR10 数据
3. 构建 ViT 模型
4. 使用交叉熵损失
5. 使用 Adam 优化器
6. 结合学习率调度器 `StepLR`
7. 训练并在测试集上评估
8. 保存模型权重到本地

## 配置说明

关键超参数都集中在 `configs/config.py`：

- `batch_size`：默认 128
- `epochs`：默认 50
- `lr`：默认 `3e-4`
- `num_workers`：数据加载线程数
- `device`：默认 `cuda`
- `log_dir`：TensorBoard 日志目录
- `model_save_path`：模型保存路径

当前保存路径为：

```text
checkpoints/cifar10_vit.pth
```

## 为什么 ViT 更难训练

这个项目本身记录了一个典型现象：

- ViT 在小数据集上通常比 CNN 更难收敛
- 对学习率、batch size 和调度器更敏感
- 训练初期波动更明显
- 更依赖数据增强和训练策略

因此在 CIFAR10 这类小图像任务上，ViT 往往需要更仔细地调参才能达到稳定结果。

## TensorBoard

训练时会把日志写入 `runs/cifar10_vit`，可使用：

```bash
tensorboard --logdir runs
```

查看 loss、accuracy 等曲线。

## 实验观察

项目中的实验记录包括：

- Patch Embedding 是否收敛顺利
- 不同学习率下的最终 accuracy
- `StepLR` 是否改善训练稳定性
- ViT 相比 CNN 的收敛速度差异

最终 `accuracy` 和训练现象在结果整理时统一补充到这一节。

## 注意事项

- 第一次运行前需要准备好 CIFAR10 数据
- 如果本地没有 `checkpoints` 目录，保存模型前需要先创建
- `test.py` 目前主要是预留文件，训练逻辑已经放在 `train.py`
