# CIFAR-10 图像分类 — 一键训练+识别入口
import os
import train
import predict

if __name__ == "__main__":
    print("=" * 50)
    print("  CIFAR-10 图像分类 — 开始训练")
    print("=" * 50)
    train.train()

    # 训练完后自动预测 predict 文件夹中的图片
    predict_dir = "predict"
    img_paths = []
    if os.path.exists(predict_dir):
        img_paths = [os.path.join(predict_dir, f)
                     for f in os.listdir(predict_dir) if f.endswith('.png')]

    print("\n" + "=" * 50)
    print("  CIFAR-10 图像分类 — 开始预测")
    print("=" * 50)
    predict.predict(img_paths)
