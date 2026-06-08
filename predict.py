# CIFAR-10 推理脚本
# 用法: python predict.py <图片路径...>
# 示例: python predict.py cat.png dog.png
import os
import sys
import torch
from PIL import Image
from torchvision import transforms

import config
from model import CIFAR10Net


def predict(img_paths):
    """对指定的图片路径列表进行预测"""
    if not img_paths:
        print("未指定图片路径")
        return

    # ========== 1. 加载模型 ==========
    model = CIFAR10Net().to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH,
                          map_location=config.DEVICE, weights_only=True))
    model.eval()

    # ========== 2. 预处理 ==========
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # ========== 3. 逐张预测 ==========
    for img_path in img_paths:
        if not os.path.exists(img_path):
            print(f"[跳过] {img_path} 不存在")
            continue
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img.to(config.DEVICE))
            pred_idx = output.argmax(dim=1).item()
            class_name = config.CLASSES[pred_idx]
            print(f"{img_path} → 预测结果: {class_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python predict.py <图片路径...>")
        print("示例: python predict.py cat.png dog.png")
        sys.exit(1)
    predict(sys.argv[1:])
