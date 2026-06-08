# CIFAR-10 超参数配置
import torch

# ========== 训练超参数 ==========
BATCH_SIZE = 256         # 批大小
LEARNING_RATE = 0.001   # 学习率
EPOCHS = 100             # 训练轮数

# ========== 图像参数 ==========
IMAGE_SIZE = 32         # CIFAR-10 图像尺寸 (32x32)
IN_CHANNELS = 3         # RGB 3通道
NUM_CLASSES = 10        # 分类数量

# CIFAR-10 类别名称
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ========== 设备 ==========
USE_GPU = True            # True: GPU训练, False: CPU训练
_GPU_AVAILABLE = torch.cuda.is_available()
if USE_GPU and _GPU_AVAILABLE:
    DEVICE = torch.device("cuda")
elif USE_GPU and not _GPU_AVAILABLE:
    print("警告: GPU 不可用，回退到 CPU")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cpu")

# ========== 路径 ==========
MODEL_SAVE_PATH = "model.pth"
DATA_DIR = "./data"
