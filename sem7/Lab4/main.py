import os
import cv2
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


dataset_path = r"C:\Users\vyvy07\PycharmProjects\SystemAIsem7_eyeclosed\IRIS + PUPIL + EYE"


def load_data(split_folder):
    images_list = []
    masks_list = []

    img_dir = os.path.join(split_folder, "image")
    seg_dir = os.path.join(split_folder, "segmentation")

   
    for filename in os.listdir(img_dir):
        if filename.endswith('.png'):
            
            img_path = os.path.join(img_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            images_list.append(img)

           
            mask_path = os.path.join(seg_dir, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            masks_list.append(mask)

    return images_list, masks_list

# 3. Загружаем train и val отдельно
print("Загружаем данные...")
train_images, train_masks = load_data(os.path.join(dataset_path, "train"))
val_images, val_masks = load_data(os.path.join(dataset_path, "val"))

print(f"Train: {len(train_images)} картинок, {len(train_masks)} масок")
print(f"Val: {len(val_images)} картинок, {len(val_masks)} масок")

# 4. Превращаем в тензоры
X_train = torch.tensor(np.array(train_images)).permute(0, 3, 1, 2).float() / 255.0
y_train = torch.tensor(np.array(train_masks)).long()

X_val = torch.tensor(np.array(val_images)).permute(0, 3, 1, 2).float() / 255.0
y_val = torch.tensor(np.array(val_masks)).long()

# Проверяем уникальные значения в масках
unique_vals = torch.unique(y_train)
print(f"Уникальные значения в масках: {unique_vals.tolist()}")

# 5. Создаем DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 6. Настраиваем устройство и модель
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Работаем на: {device}")

num_classes = len(unique_vals)
print(f"Количество классов: {num_classes}")

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes
).to(device)

# 7. Функция потерь и оптимизатор
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 8. Обучение с валидацией
epochs = 15
print("\nНачинаем обучение...")

for epoch in range(epochs):
    
    model.train()
    train_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item()

    print(f"Эпоха {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

print("\nОбучение закончено!")

# Визуализация результата на валидации
model.eval()
with torch.no_grad():
    # Берем случайную картинку из валидации
    test_idx = 0
    test_img = X_val[test_idx:test_idx+1].to(device)
    true_mask = y_val[test_idx].cpu().numpy()

    pred = model(test_img)
    pred_mask = pred.argmax(dim=1).squeeze().cpu().numpy()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Оригинал (Val)")
plt.imshow(X_val[test_idx].permute(1, 2, 0))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Истинная маска")
plt.imshow(true_mask, cmap='jet')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Предсказание модели")
plt.imshow(pred_mask, cmap='jet')
plt.axis('off')

plt.tight_layout()
plt.show()
