"""
代码使用的是零填充扩充的通道数
使用相对路径版本
"""
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
import collections  # 用于处理OrderedDict
#111
# 尝试导入 timm 库
try:
    import timm

    HAS_TIMM = True
    print("✓ timm library imported successfully.")
except ImportError:
    HAS_TIMM = False
    print("⚠️  timm 库未找到。请运行 'pip install timm' 以获取更多模型选项。")
    timm = None


# 配置参数 (Config 类)
class Config:
    # ===== 使用相对路径 =====
    train_data_root = './datas/datas1/train'
    val_data_root = './datas/datas1/val'
    output_dir = 'results/result1'

    classes = ["微带青", "杂色", "柠檬黄", "橘黄", "红棕", "青色"]
    image_size = (224, 224)
    batch_size = 32
    random_seed = 42
    max_npy_channels = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    lr_pretrained = 5e-5  # 预训练层学习率
    lr_new = 1e-3  # 新增层学习率
    weight_decay = 1e-4
    feature_dim = 512  # ResNet34 的输出特征维度 (将根据模型动态调整)
    patience = 20

    # --- 本地预训练模型权重路径（相对路径）---
    local_pretrained_paths = {
        'resnet34': './pre_models/resnet34.pth',
        'resnet50': './pre_models/resnet50.pth',
        'efficientnet_b0': './pre_models/efficientnet_b0.pth',
        'vit_base_patch16_224': './pre_models/vit_base_patch16_224.bin',
        'convnext_tiny': './pre_models/convnext_tiny.bin',
    }


# 数据转换 (get_transforms 函数)
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


# 自定义数据集 (MultimodalDataset 类)
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

        # 确保 npy_data 是 3D (H, W, C)
        if npy_data.ndim == 2:
            npy_data = np.expand_dims(npy_data, axis=-1)

        current_channels = npy_data.shape[-1]
        if current_channels < Config.max_npy_channels:
            # 填充到最大通道数
            padded_data = np.zeros((*npy_data.shape[:2], Config.max_npy_channels), dtype=np.float32)
            padded_data[:, :, :current_channels] = npy_data
            npy_data = padded_data
        elif current_channels > Config.max_npy_channels:
            # 裁剪通道
            npy_data = npy_data[:, :, :Config.max_npy_channels]

        # NPY 数据进行 Min-Max 归一化
        min_val = np.percentile(npy_data, 5)
        max_val = np.percentile(npy_data, 95)
        npy_range = max_val - min_val
        if npy_range < 1e-6:
            npy_data = np.zeros_like(npy_data) + 0.5
        else:
            npy_data = (npy_data - min_val) / npy_range
        npy_data = np.clip(npy_data, 0, 1)

        # 转换为 PyTorch tensor 并调整维度顺序 (C, H, W)
        npy_data = torch.from_numpy(npy_data).permute(2, 0, 1).float()

        if self.npy_transform:
            npy_data = self.npy_transform(npy_data)

        label = torch.tensor(label, dtype=torch.long)
        return (img.float(), npy_data), label


# 改进的多模态模型 (ImprovedMultimodalModel)
class ImprovedMultimodalModel(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet34', local_pretrained_path=None):
        super().__init__()

        self.backbone_name = backbone_name
        # 获取图像骨干网络
        self.img_backbone, self.feature_dim_img = self._get_image_backbone(backbone_name, local_pretrained_path)

        # 冻结所有层
        for param in self.img_backbone.parameters():
            param.requires_grad = False

        # 解冻特定层进行微调
        self._unfreeze_layers(backbone_name)

        # NPY 特征提取器
        self.npy_feature_extractor = nn.Sequential(
            nn.Conv2d(Config.max_npy_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, Config.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(Config.feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.feature_dim_npy = Config.feature_dim

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim_img + self.feature_dim_npy, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _get_image_backbone(self, backbone_name, local_pretrained_path):
        """
        根据名称获取图像骨干网络，并确定其输出特征维度。
        现在所有模型都从本地路径加载权重。
        """
        model = None
        feature_dim = 0

        # ===== 规范化路径（处理 Windows/Linux 路径差异）=====
        if local_pretrained_path:
            local_pretrained_path = os.path.normpath(local_pretrained_path)

        # 检查本地预训练路径是否存在
        if not local_pretrained_path or not os.path.exists(local_pretrained_path):
            abs_path = os.path.abspath(local_pretrained_path) if local_pretrained_path else "None"
            raise FileNotFoundError(
                f"未找到 '{backbone_name}' 的本地预训练权重文件\n"
                f"  相对路径: {local_pretrained_path}\n"
                f"  绝对路径: {abs_path}\n"
                f"  请确保权重文件存在于正确的位置。"
            )

        print(f"\nLoading local pretrained weights for {backbone_name}")
        print(f"  Path: {local_pretrained_path}")
        print(f"  Absolute path: {os.path.abspath(local_pretrained_path)}")

        try:
            state_dict = torch.load(local_pretrained_path, map_location='cpu')
            print(f"  ✓ Weight file loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load weight file: {e}")

        # --- 针对不同模型初始化并加载权重 ---
        if backbone_name.lower() == 'resnet34':
            model = models.resnet34(weights=None)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"  ✓ ResNet34 loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

        elif backbone_name.lower() == 'resnet50':
            model = models.resnet50(weights=None)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"  ✓ ResNet50 loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

        elif backbone_name.lower() == 'efficientnet_b0' and HAS_TIMM:
            model = timm.create_model('efficientnet_b0', pretrained=False)
            feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                name = k[6:] if k.startswith('model.') else k
                new_state_dict[name] = v
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"  ✓ EfficientNet-B0 loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

        elif backbone_name.lower() == 'vit_base_patch16_224' and HAS_TIMM:
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            feature_dim = model.head.in_features
            model.head = nn.Identity()
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                name = k[6:] if k.startswith('model.') else k
                new_state_dict[name] = v
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"  ✓ ViT-Base loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

        elif backbone_name.lower() == 'convnext_tiny' and HAS_TIMM:
            model = timm.create_model('convnext_tiny', pretrained=False)
            feature_dim = model.head.fc.in_features
            model.head.fc = nn.Identity()
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                name = k[6:] if k.startswith('model.') else k
                new_state_dict[name] = v
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"  ✓ ConvNeXt-Tiny loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

        else:
            raise ValueError(
                f"不支持的骨干网络名称: {backbone_name}。\n"
                f"请检查拼写或确保已安装 timm 库（pip install timm）。"
            )

        print(f"  Feature dimension: {feature_dim}\n")
        return model, feature_dim

    def _unfreeze_layers(self, backbone_name):
        """根据骨干网络类型解冻特定层"""
        if 'resnet' in backbone_name.lower():
            for param in self.img_backbone.layer3.parameters():
                param.requires_grad = True
            for param in self.img_backbone.layer4.parameters():
                param.requires_grad = True
            print(f"  Unfrozen: layer3 and layer4")

        elif 'efficientnet' in backbone_name.lower() and HAS_TIMM:
            for param in self.img_backbone.blocks[-2:].parameters():
                param.requires_grad = True
            print(f"  Unfrozen: last 2 blocks")

        elif ('vit' in backbone_name.lower() or 'swin' in backbone_name.lower()) and HAS_TIMM:
            for param in self.img_backbone.blocks[-4:].parameters():
                param.requires_grad = True
            if hasattr(self.img_backbone, 'norm'):
                for param in self.img_backbone.norm.parameters():
                    param.requires_grad = True
            if hasattr(self.img_backbone, 'pre_logits'):
                for param in self.img_backbone.pre_logits.parameters():
                    param.requires_grad = True
            print(f"  Unfrozen: last 4 blocks (and norm/pre_logits)")

        elif 'convnext' in backbone_name.lower() and HAS_TIMM:
            for param in self.img_backbone.stages[-2:].parameters():
                param.requires_grad = True
            print(f"  Unfrozen: last 2 stages")
        else:
            print(f"  ⚠️  No specific unfreezing strategy for {backbone_name}")

    def forward(self, x):
        img, npy = x
        img_feat = self.img_backbone(img)

        # 处理不同模型的输出格式
        if img_feat.ndim == 3 and (self.backbone_name.startswith('vit') or self.backbone_name.startswith('swin')):
            img_feat = img_feat[:, 0]  # 取 [CLS] token
        elif img_feat.ndim > 2:
            img_feat = torch.flatten(img_feat, 1)

        npy_feat = self.npy_feature_extractor(npy)
        npy_feat = torch.flatten(npy_feat, 1)

        combined = torch.cat([img_feat, npy_feat], dim=1)
        return self.classifier(combined)


# 从特定文件夹加载数据 (load_data_from_folder 函数)
def load_data_from_folder(data_dir, classes):
    """Scans a directory to find image-npy pairs for each class."""
    file_pairs = []
    labels_text = []

    # ===== 规范化路径 =====
    data_dir = os.path.normpath(data_dir)
    abs_data_dir = os.path.abspath(data_dir)

    print(f"\nScanning directory:")
    print(f"  Relative path: {data_dir}")
    print(f"  Absolute path: {abs_data_dir}")
    print(f"  Exists: {os.path.exists(data_dir)}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"数据目录不存在:\n"
            f"  相对路径: {data_dir}\n"
            f"  绝对路径: {abs_data_dir}\n"
            f"  请检查路径是否正确。"
        )

    for class_name in classes:
        class_dir = os.path.normpath(os.path.join(data_dir, class_name))
        if not os.path.isdir(class_dir):
            print(f"  ⚠️  Class directory not found: {class_name}")
            continue

        count = 0
        for file in os.listdir(class_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                base_name = os.path.splitext(file)[0]
                img_path = os.path.join(class_dir, file)
                npy_path = os.path.join(class_dir, f"{base_name}.npy")
                if os.path.exists(npy_path):
                    file_pairs.append((img_path, npy_path))
                    labels_text.append(class_name)
                    count += 1

        print(f"  ✓ {class_name}: {count} pairs")

    if not file_pairs:
        raise ValueError(
            f"未找到任何图像-npy配对文件在 {data_dir}\n"
            f"请检查:\n"
            f"  1. 数据路径是否正确\n"
            f"  2. 类别文件夹是否存在\n"
            f"  3. 图像和对应的.npy文件是否存在"
        )

    print(f"  Total: {len(file_pairs)} pairs found\n")
    return file_pairs, labels_text


# 绘制训练指标图 (plot_training_metrics 函数)
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


# 训练函数 (train_model)
def train_model(backbone_name='resnet34'):
    # ===== 规范化输出目录路径 =====
    current_output_dir = os.path.normpath(os.path.join(Config.output_dir, backbone_name))
    os.makedirs(current_output_dir, exist_ok=True)

    print(f"\n{'=' * 100}")
    print(f"Training with {backbone_name}")
    print(f"{'=' * 100}")
    print(f"Output directory:")
    print(f"  Relative: {os.path.join(Config.output_dir, backbone_name)}")
    print(f"  Absolute: {os.path.abspath(current_output_dir)}")

    train_pairs, train_labels_text = load_data_from_folder(Config.train_data_root, Config.classes)
    val_pairs, val_labels_text = load_data_from_folder(Config.val_data_root, Config.classes)

    label_encoder = LabelEncoder()
    label_encoder.fit(Config.classes)
    train_labels = label_encoder.transform(train_labels_text)
    val_labels = label_encoder.transform(val_labels_text)

    print("--- Data Summary ---")
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

    # 从 Config.local_pretrained_paths 获取当前模型的本地路径
    local_path = Config.local_pretrained_paths.get(backbone_name)

    model = ImprovedMultimodalModel(
        len(Config.classes),
        backbone_name=backbone_name,
        local_pretrained_path=local_path
    ).to(Config.device)

    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)\n")

    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-5), dtype=torch.float32).to(Config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer_grouped_parameters = [
        {'params': [p for name, p in model.img_backbone.named_parameters() if p.requires_grad],
         'lr': Config.lr_pretrained, 'weight_decay': Config.weight_decay},
        {'params': model.npy_feature_extractor.parameters(),
         'lr': Config.lr_new, 'weight_decay': Config.weight_decay},
        {'params': model.classifier.parameters(),
         'lr': Config.lr_new, 'weight_decay': Config.weight_decay}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

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

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

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
            model_save_path = os.path.join(current_output_dir, f'best_model_{backbone_name}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> ✓ Best model saved with accuracy: {100 * best_val_acc:.2f}%")
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1

        if no_improve_epoch >= Config.patience:
            print(f"\n验证准确率在 {Config.patience} 个 epoch 内没有提升。提前停止。")
            break

    print(f"\n--- Training Finished for {backbone_name} ---")
    print(f"Best validation accuracy: {100 * best_val_acc:.2f}%\n")

    history_df = pd.DataFrame(training_history)
    csv_path = os.path.join(current_output_dir, f'training_log_{backbone_name}.csv')
    history_df.to_csv(csv_path, index=False)
    print(f"✓ Training log saved: {csv_path}")

    plot_path = os.path.join(current_output_dir, f'training_metrics_{backbone_name}.png')
    plot_training_metrics(
        history_df['train_loss'], history_df['train_accuracy'],
        history_df['val_loss'], history_df['val_accuracy'],
        save_path=plot_path
    )
    print(f"✓ Training plot saved: {plot_path}")

    return best_val_acc


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)

    # ===== 预检查所有路径 =====
    print("\n" + "=" * 100)
    print("🔍 环境和路径检查")
    print("=" * 100)

    # 当前工作目录
    print(f"\n当前工作目录: {os.getcwd()}")

    # 数据路径检查
    print(f"\n数据路径:")
    for name, path in [('训练集', Config.train_data_root), ('验证集', Config.val_data_root)]:
        norm_path = os.path.normpath(path)
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        print(f"  {name}:")
        print(f"    相对路径: {norm_path}")
        print(f"    绝对路径: {abs_path}")
        print(f"    状态: {'✅ 存在' if exists else '❌ 不存在'}")

    # 权重文件检查
    print(f"\n预训练权重:")
    missing_weights = []
    for model_name, path in Config.local_pretrained_paths.items():
        norm_path = os.path.normpath(path)
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        status = '✅' if exists else '❌'
        print(f"  {status} {model_name}")
        print(f"      {norm_path}")
        if not exists:
            missing_weights.append(model_name)

    if missing_weights:
        print(f"\n⚠️  警告: 以下模型的权重文件未找到:")
        for model in missing_weights:
            print(f"    - {model}")
        print(f"  这些模型将在训练时跳过。")

    # 输出目录
    print(f"\n输出目录:")
    print(f"  相对路径: {Config.output_dir}")
    print(f"  绝对路径: {os.path.abspath(Config.output_dir)}")

    print("=" * 100 + "\n")

    # ===== 开始模型对比训练 =====
    models_to_compare = [
        'resnet34',
        'resnet50',
        'efficientnet_b0',
        'vit_base_patch16_224',
        'convnext_tiny'
    ]

    results = {}
    for model_name in models_to_compare:
        # 检查是否需要 timm
        if model_name in ['efficientnet_b0', 'vit_base_patch16_224', 'convnext_tiny']:
            if not HAS_TIMM:
                print(f"⚠️  Skipping {model_name} (timm not installed)")
                results[model_name] = "timm Not Installed"
                continue

        # 检查权重文件是否存在
        if model_name in missing_weights:
            print(f"⚠️  Skipping {model_name} (weight file not found)")
            results[model_name] = "Weight File Not Found"
            continue

        try:
            best_acc = train_model(backbone_name=model_name)
            results[model_name] = best_acc
        except FileNotFoundError as e:
            print(f"❌ Error training {model_name}: {e}")
            results[model_name] = "File Not Found"
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            import traceback

            traceback.print_exc()
            results[model_name] = "Error"

        print("-" * 100)

    # ===== 打印对比结果 =====
    print("\n" + "=" * 100)
    print("📊 模型对比结果")
    print("=" * 100)
    for model_name, acc in results.items():
        if isinstance(acc, float):
            print(f"  {model_name:30s}: {100 * acc:.2f}%")
        else:
            print(f"  {model_name:30s}: {acc}")
    print("=" * 100 + "\n")