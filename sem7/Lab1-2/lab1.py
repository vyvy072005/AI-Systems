# Лабораторная работа 1: классификация глаз водителя (открыты / закрыты)
# Датасет: MRL Eye Dataset, https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset
import os
import random

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Переходим в папку со скриптом, чтобы пути data/... работали при запуске из любой папки
os.chdir(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = "data/train"  # внутри папки классов: "close eyes", "open eyes"
TEST_DIR = "data/test"
EPOCHS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Устройство:", device)

# 1. Загружаем картинки с диска. Имя папки = название класса.
transform = transforms.Compose([
    transforms.Grayscale(),        # снимки инфракрасные, цвет не нужен
    transforms.Resize((64, 64)),   # все картинки к одному размеру
    transforms.ToTensor(),         # картинка -> тензор 1x64x64 со значениями 0..1
])

# Используем одинаковый transform без аугментации для обоих наборов данных
train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_data = datasets.ImageFolder(TEST_DIR, transform=transform)
classes = train_data.classes
print("Классы:", classes, "| train:", len(train_data), "| test:", len(test_data))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 2. Модель: маленькая свёрточная сеть.
# Каждый блок Conv -> ReLU -> MaxPool уменьшает картинку вдвое: 64 -> 32 -> 16 -> 8
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),                       # 64 канала * 8 * 8 -> вектор
    nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
    nn.Dropout(0.5),                    # против переобучения
    nn.Linear(128, len(classes)),       # 2 выхода: по одному на класс
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def test_accuracy():
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(test_data)


# 3. Обучение
history = {"loss": [], "train_acc": [], "test_acc": []}
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()

    history["loss"].append(total_loss / len(train_data))
    history["train_acc"].append(correct / len(train_data))
    history["test_acc"].append(test_accuracy())
    print(f"Эпоха {epoch + 1}/{EPOCHS}: loss={history['loss'][-1]:.4f} "
          f"train_acc={history['train_acc'][-1]:.3f} test_acc={history['test_acc'][-1]:.3f}")

torch.save(model.state_dict(), "model.pth")

# 4. Визуализация: графики обучения
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history["loss"])
plt.title("Функция потерь (train)")
plt.xlabel("эпоха")
plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="train")
plt.plot(history["test_acc"], label="test")
plt.title("Точность")
plt.xlabel("эпоха")
plt.legend()
plt.savefig("training.png")

# 5. Визуализация: примеры классификации на тестовых картинках
# Зелёная подпись - модель угадала, красная - ошиблась
model.eval()
plt.figure(figsize=(10, 10))
for i, idx in enumerate(random.sample(range(len(test_data)), 16)):
    img, label = test_data[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    plt.subplot(4, 4, i + 1)
    plt.imshow(img[0], cmap="gray")
    plt.title(f"прогноз: {classes[pred]}\nверно: {classes[label]}",
              color="green" if pred == label else "red", fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig("examples1.png")
plt.show()