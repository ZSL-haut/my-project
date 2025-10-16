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
from torch.optim.lr_scheduler import CosineAnnealingLR

# 配置参数
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
    output_dir = '../results/result1/resnet50'
    local_resnet_pretrained_path = '../pre_models/resnet50.pth'


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


# 自定义数据集
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
        if npy_data.ndim == 2:
            npy_data = np.expand_dims(npy_data, axis=-1)

        current_channels = npy_data.shape[-1]
        if current_channels < Config.max_npy_channels:
            padded_data = np.zeros((*npy_data.shape[:2], Config.max_npy_channels), dtype=np.float32)
            padded_data[:, :, :current_channels] = npy_data
            npy_data = padded_data
        elif current_channels > Config.max_npy_channels:
            npy_data = npy_data[:, :, :Config.max_npy_channels]

        # Min-Max Normalization
        min_val = np.percentile(npy_data, 5)
        max_val = np.percentile(npy_data, 95)
        npy_range = max_val - min_val
        if npy_range < 1e-6:
            npy_data = np.zeros_like(npy_data) + 0.5
        else:
            npy_data = (npy_data - min_val) / npy_range
        npy_data = np.clip(npy_data, 0, 1)

        npy_data = torch.from_numpy(npy_data).permute(2, 0, 1).float()

        if self.npy_transform:
            npy_data = self.npy_transform(npy_data)

        label = torch.tensor(label, dtype=torch.long)
        return (img.float(), npy_data), label


def load_resnet_pretrained_weights(model, weight_path):
    """
    加载 ResNet50 预训练权重的专用函数
    支持 .pth 和 .bin 格式，处理键名不匹配问题

    Args:
        model: PyTorch ResNet50 模型实例
        weight_path: 权重文件路径

    Returns:
        model: 加载权重后的模型
    """
    print(f"\n{'=' * 100}")
    print(f"正在加载 ResNet50 预训练权重...")
    print(f"权重文件路径: {weight_path}")
    print(f"{'=' * 100}")

    if not os.path.exists(weight_path):
        print(f"⚠️  警告：权重文件不存在")
        print(f"   路径: {weight_path}")
        print(f"   将使用随机初始化或在线下载权重")
        return None

    try:
        # 1. 加载权重文件
        print(f"正在读取权重文件...")
        checkpoint = torch.load(weight_path, map_location='cpu')
        print(f"✓ 权重文件加载成功")

        # 2. 提取 state_dict
        if isinstance(checkpoint, dict):
            # 尝试常见的键名
            possible_keys = ['model', 'state_dict', 'model_state', 'net']
            state_dict = None

            for key in possible_keys:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"✓ 从 checkpoint['{key}'] 提取 state_dict")
                    break

            if state_dict is None:
                # 假设 checkpoint 本身就是 state_dict
                state_dict = checkpoint
                print(f"✓ checkpoint 本身就是 state_dict")
        else:
            state_dict = checkpoint
            print(f"✓ checkpoint 本身就是 state_dict")

        # 3. 打印权重信息
        print(f"\n权重文件信息:")
        print(f"  - 包含参数数量: {len(state_dict)}")
        sample_keys = list(state_dict.keys())[:5]
        print(f"  - 示例键名: {sample_keys}")

        # 4. 获取模型的 state_dict
        model_state_dict = model.state_dict()
        model_keys = list(model_state_dict.keys())
        print(f"\n模型信息:")
        print(f"  - 模型参数数量: {len(model_keys)}")
        print(f"  - 示例键名: {model_keys[:5]}")

        # 5. 尝试加载（允许部分匹配）
        print(f"\n{'─' * 100}")
        print(f"开始加载权重...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # 6. 统计匹配情况
        total_keys = len(model_keys)
        matched_keys = total_keys - len(missing_keys)
        match_ratio = (matched_keys / total_keys) * 100

        print(f"\n{'─' * 100}")
        print(f"📊 权重加载统计:")
        print(f"  - 模型总参数: {total_keys}")
        print(f"  - ✅ 成功匹配: {matched_keys} ({match_ratio:.2f}%)")
        print(f"  - ❌ 缺失的键: {len(missing_keys)}")
        print(f"  - ⚠️  多余的键: {len(unexpected_keys)}")

        # 7. 根据匹配率给出诊断
        if match_ratio >= 95:
            print(f"\n✅ 预训练权重加载完美！匹配率: {match_ratio:.2f}%")
            print(f"   模型已成功加载预训练权重，可以开始训练")
        elif match_ratio >= 80:
            print(f"\n✅ 预训练权重加载成功！匹配率: {match_ratio:.2f}%")
            print(f"   大部分权重已加载，少数层使用随机初始化")
        elif match_ratio >= 50:
            print(f"\n⚠️  警告：预训练权重部分加载，匹配率: {match_ratio:.2f}%")
            print(f"   约一半的权重被加载，建议检查权重文件")
        else:
            print(f"\n❌ 错误：权重匹配率过低 ({match_ratio:.2f}%)")
            print(f"   权重文件可能不兼容，建议：")
            print(f"   1. 检查权重文件是否是 ResNet50")
            print(f"   2. 确认权重来源（torchvision/其他）")
            print(f"   3. 考虑使用 torchvision 官方预训练权重")

        # 8. 如果缺失键较少，打印详情
        if 0 < len(missing_keys) <= 20:
            print(f"\n缺失的键（这些层将使用随机初始化）:")
            for i, key in enumerate(missing_keys[:10], 1):
                print(f"   {i}. {key}")
            if len(missing_keys) > 10:
                print(f"   ... 还有 {len(missing_keys) - 10} 个")

        # 9. 如果有多余的键且数量不多，打印部分
        if 0 < len(unexpected_keys) <= 20:
            print(f"\n多余的键（权重文件中存在但模型不需要）:")
            for i, key in enumerate(unexpected_keys[:10], 1):
                print(f"   {i}. {key}")
            if len(unexpected_keys) > 10:
                print(f"   ... 还有 {len(unexpected_keys) - 10} 个")

        print(f"{'=' * 100}\n")
        return model

    except Exception as e:
        print(f"\n❌ 加载权重失败: {e}")
        print(f"错误详情:")
        import traceback
        traceback.print_exc()
        print(f"\n将使用随机初始化或在线下载权重继续")
        print(f"{'=' * 100}\n")
        return None


# 改进的多模态模型
class ImprovedMultimodalResNet(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super().__init__()

        print(f"\n{'=' * 100}")
        print("初始化 ResNet50 多模态模型")
        print(f"{'=' * 100}\n")

        # 1. 创建 ResNet50 模型（不加载预训练权重）
        print("创建 ResNet50 模型结构...")
        self.img_resnet = models.resnet50(weights=None)
        print("✓ 模型结构创建成功")

        # 2. 尝试加载本地预训练权重
        weights_loaded = False
        if pretrained_path and os.path.exists(pretrained_path):
            result = load_resnet_pretrained_weights(self.img_resnet, pretrained_path)
            if result is not None:
                weights_loaded = True

        # 3. 如果本地权重加载失败，尝试在线下载
        if not weights_loaded:
            print(f"\n{'=' * 100}")
            print("⚠️  本地预训练权重未成功加载")
            if pretrained_path:
                print(f"   路径: {pretrained_path}")
            print("   尝试使用 torchvision 在线预训练权重...")
            try:
                self.img_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                print("✅ 使用 torchvision ImageNet 预训练权重")
                weights_loaded = True
            except Exception as e:
                print(f"❌ 在线下载也失败: {e}")
                print("   将使用随机初始化（不推荐，训练效果会较差）")
            print(f"{'=' * 100}\n")

        # 4. 配置参数冻结策略
        print("配置模型参数冻结策略...")
        total_params = sum(p.numel() for p in self.img_resnet.parameters())

        # 冻结大部分参数
        for param in self.img_resnet.parameters():
            param.requires_grad = False

        # 解冻 layer4（最后一个残差块）
        for param in self.img_resnet.layer4.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.img_resnet.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"✓ ResNet50 参数配置:")
        print(f"  - 总参数: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")
        print(f"  - 冻结参数: {frozen_params:,} ({frozen_params / total_params * 100:.1f}%)")

        # 5. 替换分类头
        num_ftrs = self.img_resnet.fc.in_features
        self.img_resnet.fc = nn.Identity()
        print(f"✓ ResNet50 特征维度: {num_ftrs}")

        # 6. NPY 特征提取器
        self.npy_feature_extractor = nn.Sequential(
            nn.Conv2d(Config.max_npy_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        print(f"✓ NPY 特征提取器创建成功 (输出维度: 128)")

        # 7. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        npy_params = sum(p.numel() for p in self.npy_feature_extractor.parameters())

        print(f"✓ 多模态融合分类器创建成功")
        print(f"  - 输入维度: {num_ftrs + 128} (图像 {num_ftrs} + NPY 128)")
        print(f"  - 输出类别数: {num_classes}")
        print(f"  - 分类器参数: {classifier_params:,}")
        print(f"  - NPY 提取器参数: {npy_params:,}")
        print(f"\n{'=' * 100}\n")

    def forward(self, x):
        img, npy = x
        img_feat = self.img_resnet(img)
        npy_feat = self.npy_feature_extractor(npy)
        npy_feat = torch.flatten(npy_feat, 1)
        combined = torch.cat([img_feat, npy_feat], dim=1)
        return self.classifier(combined)


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
        raise ValueError(f"No matching image/npy file pairs found in {data_dir}.")
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
    npy_transform = transforms.Compose([
        transforms.Resize(Config.image_size, antialias=True)
    ])

    train_data = [(p[0], p[1], lbl) for p, lbl in zip(train_pairs, train_labels)]
    val_data = [(p[0], p[1], lbl) for p, lbl in zip(val_pairs, val_labels)]

    train_dataset = MultimodalDataset(train_data, transform=train_transform, npy_transform=npy_transform)
    val_dataset = MultimodalDataset(val_data, transform=test_transform, npy_transform=npy_transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, num_workers=4, pin_memory=True)

    # ===== 修改这里：使用正确的预训练模型路径 =====
    model = ImprovedMultimodalResNet(
        len(Config.classes),
        pretrained_path=Config.local_resnet_pretrained_path  # 使用 Config 中定义的路径
    ).to(Config.device)

    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-5), dtype=torch.float32).to(Config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.img_resnet.parameters() if p.requires_grad],
         'lr': Config.lr_pretrained, 'weight_decay': Config.weight_decay},
        {'params': model.npy_feature_extractor.parameters(),
         'lr': Config.lr_new, 'weight_decay': Config.weight_decay},
        {'params': model.classifier.parameters(),
         'lr': Config.lr_new, 'weight_decay': Config.weight_decay}
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.num_epochs, eta_min=1e-7)

    training_history = []
    best_val_acc = 0
    no_improve_epoch = 0

    print("=" * 100)
    print("--- Starting Training ---")
    print("=" * 100 + "\n")

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

        scheduler.step()

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
            torch.save(model.state_dict(), os.path.join(Config.output_dir, 'model.pth'))
            print(f"  -> New best model saved with accuracy: {100 * best_val_acc:.2f}%")
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1

        if no_improve_epoch >= Config.patience:
            print(f"\nValidation accuracy did not improve for {Config.patience} epochs. Early stopping.")
            break

    print("\n" + "=" * 100)
    print("--- Training Finished ---")
    print("=" * 100 + "\n")

    history_df = pd.DataFrame(training_history)
    csv_path = os.path.join(Config.output_dir, 'training_log_resnet50.csv')
    history_df.to_csv(csv_path, index=False)
    print(f"Training log saved to: {csv_path}")

    plot_path = os.path.join(Config.output_dir, 'training_metrics_resnet50.png')
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