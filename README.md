# CIFAR-10 图像分类

基于 PyTorch 的 CIFAR-10 图像分类项目，使用 CNN 卷积神经网络（含 BatchNorm + Dropout）实现 10 类彩色图像的自动分类。

## 项目结构

```
final_experiment/
├── main.py              # 入口：一键训练 + 识别
├── config.py            # 超参数配置（学习率、批次、GPU/CPU切换等）
├── model.py             # CNN 模型定义（CIFAR10Net）
├── train.py             # 训练脚本（数据增强 + 训练 + 验证 + 保存模型）
├── predict.py           # 推理脚本（命令行传入图片路径即可预测）
├── setup_data.py        # 数据准备（下载 CIFAR-10 → 7:3拆分 → 保存为PNG）
├── data/                # 数据集（训练集/测试集，按类别分文件夹）
│   ├── train/           # 训练集 35,000 张
│   └── test/            # 测试集 15,000 张
├── predict/             # 示例预测图片
└── .gitignore           # Git 忽略规则（排除 data/、model.pth、图片等）
```

## 环境依赖

- Python 3.9+
- PyTorch >= 2.0
- torchvision
- Pillow

```bash
pip install torch torchvision pillow
```

## 快速开始

### 1. 准备数据

首次运行需要准备 CIFAR-10 数据集：

```bash
python setup_data.py
```

该脚本会下载 CIFAR-10 数据集，按 **7:3** 的比例拆分为训练集（35,000 张）和测试集（15,000 张），以 PNG 格式存入 `data/train/` 和 `data/test/`。

### 2. 训练 + 预测

```bash
python main.py
```

一键完成训练和预测，训练完成后自动对 `predict/` 文件夹中的图片进行分类。

### 3. 单独预测

将任意 32×32 以上尺寸的 RGB 图片放在项目文件夹下，运行：

```bash
python predict.py cat.png
python predict.py img1.png img2.png img3.png
```

## 模型架构

| 层 | 参数 | 输出尺寸 |
|----|------|----------|
| Conv2D + BN + ReLU | 3→32, k=3, pad=1 | 32×32 |
| Conv2D + BN + ReLU | 32→64, k=3, pad=1 | 32×32 |
| MaxPool2D | k=2, stride=2 | 16×16 |
| Conv2D + BN + ReLU | 64→128, k=3, pad=1 | 16×16 |
| Conv2D + BN + ReLU | 128→128, k=3, pad=1 | 16×16 |
| MaxPool2D | k=2, stride=2 | 8×8 |
| Flatten | — | 8192 |
| Linear + ReLU + Dropout(0.5) | 8192→256 | 256 |
| Linear + LogSoftmax | 256→10 | 10 |

**特点**：BatchNorm 加速收敛 + Dropout 防止过拟合 + 数据增强（RandomCrop + RandomHorizontalFlip）

## 配置说明（config.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 256 | 批大小 |
| `LEARNING_RATE` | 0.001 | 学习率 |
| `EPOCHS` | 100 | 训练轮数 |
| `USE_GPU` | `True` | `True`: GPU, `False`: CPU |
| `CLASSES` | 10类 | CIFAR-10 类别名称列表 |

**GPU/CPU 切换**：只需在 `config.py` 中修改 `USE_GPU` 即可，无需改动其他任何代码。

## 训练结果

使用 CPU 训练 20 轮的结果（测试集准确率）：

| Epoch | Test Acc |
|-------|----------|
| 1 | 40.71% |
| 5 | 59.82% |
| 10 | 70.70% |
| 15 | 72.34% |
| 20 | **76.73%** |

> 使用 GPU 并将 EPOCHS 设为 100，预期准确率可达 85%+。

## License

MIT
