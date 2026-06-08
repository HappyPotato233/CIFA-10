# 数据准备脚本：下载 CIFAR-10 → 拆分为 train/test → 保存为图片
import os
import shutil
from PIL import Image
from torchvision import datasets

import config

TRAIN_DIR = os.path.join(config.DATA_DIR, "train")
TEST_DIR = os.path.join(config.DATA_DIR, "test")
PREDICT_DIR = "predict"
SPLIT_RATIO = 0.7  # 70% 训练，30% 测试


def setup():
    # 清理旧数据
    for d in [TRAIN_DIR, TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(PREDICT_DIR, exist_ok=True)

    # 下载 CIFAR-10
    print("下载 CIFAR-10 数据集...")
    dataset = datasets.CIFAR10(root="./cifar10_raw", train=True, download=False)

    # 为每个类别创建文件夹
    for cls_name in config.CLASSES:
        os.makedirs(os.path.join(TRAIN_DIR, cls_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, cls_name), exist_ok=True)

    # 按类别分组
    class_images = {cls: [] for cls in config.CLASSES}
    for idx, (img, label) in enumerate(dataset):
        class_images[config.CLASSES[label]].append((idx, img))

    # 按 7:3 拆分并保存
    total_train = 0
    total_test = 0
    predict_candidates = {}
    wanted_predict = ['airplane', 'dog', 'ship']

    for cls_name, images in class_images.items():
        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        for idx, img in train_imgs:
            img.save(os.path.join(TRAIN_DIR, cls_name, f"{idx}.png"))
        for idx, img in test_imgs:
            img.save(os.path.join(TEST_DIR, cls_name, f"{idx}.png"))

        total_train += len(train_imgs)
        total_test += len(test_imgs)

        # 保存 predict 候选图（每个类别第一张）
        if cls_name in wanted_predict:
            predict_candidates[cls_name] = images[0][1]

    # 保存 predict 图片
    for cls_name, img in predict_candidates.items():
        img = img.resize((280, 280), Image.NEAREST)
        img.save(os.path.join(PREDICT_DIR, f"{cls_name}.png"))

    print(f"训练集: {total_train} 张, 测试集: {total_test} 张")
    print(f"predict 图片: {list(predict_candidates.keys())}")
    print("数据准备完成！")

    # 清理原始下载的缓存
    shutil.rmtree("./cifar10_raw", ignore_errors=True)


if __name__ == "__main__":
    setup()
