import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


# Configuration Parameters
class Config:
    train_data_root = '../datas/datas1/train'
    val_data_root = '../datas/datas1/val'
    classes = ["微带青", "杂色", "柠檬黄", "橘黄", "红棕", "青色"]
    image_size = (224, 224)
    batch_size = 32
    random_seed = 42
    max_npy_channels = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    lr_pretrained = 5e-5
    lr_new = 1e-3
    weight_decay = 1e-4
    feature_dim = 512
    patience = 20
    output_dir = '../results/result1/resnet34'
    local_resnet_pretrained_path = '../pre_models/resnet34.pth'


# Data Transforms
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform


# Custom Dataset
class MultimodalDataset(Dataset):
    def __init__(self, file_pairs, transform=None, npy_transform=None):
        self.file_pairs = file_pairs
        self.transform = transform
        self.npy_transform = npy_transform

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_path, npy_path, label = self.file_pairs[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        npy_data = np.load(npy_path).astype(np.float32)

        # Ensure npy_data is 3D (H, W, C)
        if npy_data.ndim == 2:
            npy_data = np.expand_dims(npy_data, axis=-1)

        current_channels = npy_data.shape[-1]
        if current_channels < Config.max_npy_channels:
            # Pad to max channels
            padded_data = np.zeros((*npy_data.shape[:2], Config.max_npy_channels), dtype=np.float32)
            padded_data[:, :, :current_channels] = npy_data
            npy_data = padded_data
        elif current_channels > Config.max_npy_channels:
            # Crop channels
            npy_data = npy_data[:, :, :Config.max_npy_channels]

        # Min-Max Normalization for NPY data
        min_val = np.percentile(npy_data, 5)
        max_val = np.percentile(npy_data, 95)
        npy_range = max_val - min_val
        if npy_range < 1e-6:
            npy_data = np.zeros_like(npy_data) + 0.5
        else:
            npy_data = (npy_data - min_val) / npy_range
        npy_data = np.clip(npy_data, 0, 1)

        # Convert to PyTorch tensor and adjust dimension order (C, H, W)
        npy_data = torch.from_numpy(npy_data).permute(2, 0, 1).float()

        if self.npy_transform:
            npy_data = self.npy_transform(npy_data)

        label = torch.tensor(label, dtype=torch.long)
        return (img.float(), npy_data), label


# Improved Multimodal ResNet Model
class ImprovedMultimodalResNet(nn.Module):
    def __init__(self, num_classes, pretrained_path=None): # Added pretrained_path argument
        super().__init__()

        # Use ResNet34
        self.img_resnet = models.resnet34(weights=None) # Initialize without pre-trained ImageNet weights
        #加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading local pretrained ResNet34 weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu') # Load to CPU first
            self.img_resnet.load_state_dict(state_dict)
        else:
            print("Warning: Local pretrained ResNet34 weights not found or path not specified. "
                  "Initializing ResNet34 with default ImageNet weights.")
            self.img_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)


        # 冻结所有层
        for param in self.img_resnet.parameters():
            param.requires_grad = False
        # 解冻 layer3 and layer4 (有助于微调)
        for param in self.img_resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.img_resnet.layer4.parameters():
            param.requires_grad = True

        #替换 ResNet 网络中的最终全连接层（fc）为 nn.Identity() 模块，从而使得模型不进行最终的分类，
        #而是将图像的特征从网络的最后一层卷积提取出来。
        self.img_resnet.fc = nn.Identity()   #恒等模块，不执行任何操作

        # NPY Feature Extractor
        self.npy_feature_extractor = nn.Sequential(
            nn.Conv2d(Config.max_npy_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, Config.feature_dim, kernel_size=3, padding=1), # Output to match ResNet34's feature dim
            nn.BatchNorm2d(Config.feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  #自适应池化会根据输入的尺寸自动调整池化的区域大小，以确保输出特征图的空间尺寸是预定的大小。
        )

        # Classifier
        # ResNet34's feature dimension (before fc layer) is 512
        # NPY feature extractor outputs Config.feature_dim (which is 512)
        self.classifier = nn.Sequential(
            nn.Linear(Config.feature_dim + Config.feature_dim, 512), # 512 (img) + 512 (npy)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        img, npy = x
        img_feat = self.img_resnet(img)
        npy_feat = self.npy_feature_extractor(npy)
        npy_feat = torch.flatten(npy_feat, 1)
        combined = torch.cat([img_feat, npy_feat], dim=1)
        return self.classifier(combined)


# Function to load data from a specific folder
def load_data_from_folder(data_dir, classes):
    """Scans a directory to find image-npy pairs for each class."""
    file_pairs = []
    labels_text = []
    print(f"Scanning directory: {data_dir}")
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory not found, skipping: {class_dir}")
            continue
        for file in os.listdir(class_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                base_name = os.path.splitext(file)[0]
                img_path = os.path.join(class_dir, file)
                npy_path = os.path.join(class_dir, f"{base_name}.npy")
                if os.path.exists(npy_path):
                    file_pairs.append((img_path, npy_path))
                    labels_text.append(class_name)
    if not file_pairs:
        raise ValueError(
            f"No matching image/npy file pairs found in {data_dir}. Please check your data paths and structure.")
    return file_pairs, labels_text


def plot_training_metrics(train_losses, train_accs, val_losses, val_accs, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Training Function
def train_model():
    os.makedirs(Config.output_dir, exist_ok=True)

    train_pairs, train_labels_text = load_data_from_folder(Config.train_data_root, Config.classes)
    val_pairs, val_labels_text = load_data_from_folder(Config.val_data_root, Config.classes)

    label_encoder = LabelEncoder()
    label_encoder.fit(Config.classes)
    train_labels = label_encoder.transform(train_labels_text)
    val_labels = label_encoder.transform(val_labels_text)

    print("\n--- Data Summary ---")
    print(f"Training samples: {len(train_labels)}")
    print("Training class distribution:", Counter(train_labels_text))
    print(f"Validation samples: {len(val_labels)}")
    print("Validation class distribution:", Counter(val_labels_text))
    print("---------------------\n")

    train_transform, test_transform = get_transforms()

    npy_train_transform = transforms.Compose([
        transforms.Resize(Config.image_size, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    npy_test_transform = transforms.Compose([
        transforms.Resize(Config.image_size, antialias=True)
    ])

    train_data = [(p[0], p[1], lbl) for p, lbl in zip(train_pairs, train_labels)]
    val_data = [(p[0], p[1], lbl) for p, lbl in zip(val_pairs, val_labels)]

    train_dataset = MultimodalDataset(train_data, transform=train_transform, npy_transform=npy_train_transform)
    val_dataset = MultimodalDataset(val_data, transform=test_transform, npy_transform=npy_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, num_workers=4, pin_memory=True)

    # Pass the local pre-trained path to the model constructor
    model = ImprovedMultimodalResNet(len(Config.classes), pretrained_path=Config.local_resnet_pretrained_path).to(Config.device)

    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-5), dtype=torch.float32).to(Config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW([
        {'params': [p for name, p in model.img_resnet.named_parameters() if p.requires_grad],
         'lr': Config.lr_pretrained, 'weight_decay': Config.weight_decay},
        {'params': model.npy_feature_extractor.parameters(), 'lr': Config.lr_new, 'weight_decay': Config.weight_decay},
        {'params': model.classifier.parameters(), 'lr': Config.lr_new, 'weight_decay': Config.weight_decay}
    ])

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=Config.patience // 2, verbose=True)

    training_history = []
    best_val_acc = 0
    no_improve_epoch = 0

    print("--- Starting Training ---")
    for epoch in range(Config.num_epochs):
        model.train()
        total_loss, correct, total_samples = 0, 0, 0
        for batch_idx, ((img, npy), labels) in enumerate(train_loader):
            img, npy, labels = img.to(Config.device), npy.to(Config.device), labels.to(Config.device)
            optimizer.zero_grad()
            outputs = model((img, npy))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += img.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = correct / total_samples

        model.eval()
        val_correct, val_total_loss, val_total_samples = 0, 0, 0
        with torch.no_grad():
            for (img, npy), labels in val_loader:
                img, npy, labels = img.to(Config.device), npy.to(Config.device), labels.to(Config.device)
                outputs = model((img, npy))
                val_total_loss += criterion(outputs, labels).item() * img.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total_samples += img.size(0)

        val_loss = val_total_loss / val_total_samples
        val_acc = val_correct / val_total_samples

        scheduler.step(val_acc) # Step the scheduler with validation accuracy

        current_lr = optimizer.param_groups[1]['lr']

        print(f"Epoch {epoch + 1}/{Config.num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {100 * epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {100 * val_acc:.2f}% | "
              f"LR: {current_lr:.2e}")

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': round(epoch_loss, 5),
            'train_accuracy': round(epoch_acc, 5),
            'val_loss': round(val_loss, 5),
            'val_accuracy': round(val_acc, 5),
            'learning_rate': current_lr
        }
        training_history.append(log_entry)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(Config.output_dir, 'best_model_resnet34.pth'))
            print(f"  -> New best model saved with accuracy: {100 * best_val_acc:.2f}%")
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1

        if no_improve_epoch >= Config.patience:
            print(f"\nValidation accuracy did not improve for {Config.patience} epochs. Early stopping.")
            break

    print("--- Training Finished ---")

    history_df = pd.DataFrame(training_history)
    csv_path = os.path.join(Config.output_dir, 'training_log_resnet34.csv')
    history_df.to_csv(csv_path, index=False)
    print(f"Training log saved to: {csv_path}")

    plot_path = os.path.join(Config.output_dir, 'training_metrics_resnet34.png')
    plot_training_metrics(
        history_df['train_loss'], history_df['train_accuracy'],
        history_df['val_loss'], history_df['val_accuracy'],
        save_path=plot_path
    )
    print(f"Training plot saved to: {plot_path}")


if __name__ == "__main__":
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    train_model()