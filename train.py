# CIFAR-10 训练脚本
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
from model import CIFAR10Net


def train():
    # ========== 1. 数据增强与加载 ==========
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.ImageFolder(
        root=f"{config.DATA_DIR}/train", transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=f"{config.DATA_DIR}/test", transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    # ========== 2. 模型、优化器、调度器 ==========
    model = CIFAR10Net().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # ========== 3. 训练循环 ==========
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_train_loss = 0
        correct_train = 0

        for data, target in train_loader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_train += pred.eq(target).sum().item()

        train_acc = 100.0 * correct_train / len(train_loader.dataset)
        scheduler.step()

        # ========== 4. 验证 ==========
        model.eval()
        total_test_loss = 0
        correct_test = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                output = model(data)
                total_test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                correct_test += pred.eq(target).sum().item()

        test_loss = total_test_loss / len(test_loader.dataset)
        test_acc = 100.0 * correct_test / len(test_loader.dataset)

        print(f"Epoch {epoch:2d}/{config.EPOCHS}  "
              f"Train Loss: {total_train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%")

    # ========== 5. 保存模型 ==========
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"\n模型已保存至: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
