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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import glob

# 使用Agg后端以支持非交互式环境下的绘图
matplotlib.use('Agg')

# 尝试导入 timm 库
try:
    import timm
except ImportError:
    timm = None


class Config:
    test_data_root = r"F:\烟草项目二期\datas6\test"
    classes = ["浓", "强", "中", "弱", "淡"]
    image_size = (224, 224)
    batch_size = 32
    random_seed = 42
    max_npy_channels = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = 512
    # 结果保存路径
    results_save_dir = './results/result6/unified_metrics'
    # 训练好的模型权重路径
    best_model_paths = {
        'convnext_tiny': './results/result6/convnext_tiny/best_model_convnext_tiny.pth',
        'efficientnet_b0': './results/result6/efficientnet_b0/best_model_efficientnet_b0.pth',
        'resnet34': './results/result6/ResNet34/best_model_resnet34.pth',
        'resnet50': './results/result6/ResNet50/best_model_resnet50.pth',
        'vit_base_patch16_224': './results/result6/vit_base_patch16_224/best_model_vit_base_patch16_224.pth',
        'swin_tiny_patch4_window7_224': './results/result6/swin_tiny_patch4_window7_224/best_model_swin_tiny_patch4_window7_224.pth'
    }

    # 预训练模型路径
    local_pretrained_paths = {
        'resnet34': "./pre_models/resnet34.pth",
        'resnet50': "./pre_models/resnet50.pth",
        'efficientnet_b0': "./pre_models/efficientnet_b0.bin",
        'vit_base_patch16_224': "./pre_models/vit_base_patch16_224.bin",
        'swin_tiny_patch4_window7_224': "./pre_models/swin_tiny_patch4_window7_224.bin",
        'convnext_tiny': "./pre_models/convnext_tiny.bin",
    }


# 创建结果保存目录
os.makedirs(Config.results_save_dir, exist_ok=True)


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


class ImprovedMultimodalModel(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet34', local_pretrained_path=None):
        super().__init__()
        self.backbone_name = backbone_name
        self.img_backbone, self.feature_dim_img = self._get_image_backbone(backbone_name, local_pretrained_path)
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
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim_img + self.feature_dim_npy, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _get_image_backbone(self, backbone_name, local_pretrained_path):
        model = None
        feature_dim = 0

        if backbone_name.lower() == 'resnet34':
            model = models.resnet34(weights=None)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif backbone_name.lower() == 'resnet50':
            model = models.resnet50(weights=None)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif backbone_name.lower() == 'efficientnet_b0' and timm:
            model = timm.create_model('efficientnet_b0', pretrained=False)
            feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()
        elif backbone_name.lower() == 'vit_base_patch16_224' and timm:
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            feature_dim = model.head.in_features
            model.head = nn.Identity()
        elif backbone_name.lower() == 'swin_tiny_patch4_window7_224' and timm:
            model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
            feature_dim = model.head.in_features
            model.head = nn.Identity()
        elif backbone_name.lower() == 'convnext_tiny' and timm:
            model = timm.create_model('convnext_tiny', pretrained=False)
            feature_dim = model.head.fc.in_features
            model.head.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone name: {backbone_name}")

        return model, feature_dim

    def forward(self, x):
        img, npy = x
        img_feat = self.img_backbone(img)
        if img_feat.ndim == 3 and (self.backbone_name.startswith('vit') or self.backbone_name.startswith('swin')):
            img_feat = img_feat[:, 0]
        elif img_feat.ndim > 2:
            img_feat = torch.flatten(img_feat, 1)

        npy_feat = self.npy_feature_extractor(npy)
        npy_feat = torch.flatten(npy_feat, 1)

        combined = torch.cat([img_feat, npy_feat], dim=1)
        return self.classifier(combined)


def load_data_from_folder(data_dir, classes):
    """
    Scans a directory to find image-npy pairs for each class.
    增强版：提供详细的调试信息，处理路径编码问题
    """
    file_pairs = []
    labels_text = []

    print(f"\n{'=' * 80}")
    print(f"正在扫描测试数据目录: {data_dir}")
    print(f"{'=' * 80}")

    # 首先检查根目录是否存在
    if not os.path.exists(data_dir):
        raise ValueError(f"❌ 错误：测试数据根目录不存在: {data_dir}")

    # 获取目录下的所有子目录
    try:
        all_subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"✓ 找到 {len(all_subdirs)} 个子目录: {all_subdirs}")
    except Exception as e:
        raise ValueError(f"❌ 无法读取目录 {data_dir}: {e}")

    # 检查每个期望的类目录
    found_classes = []
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)

        print(f"\n{'─' * 80}")
        print(f"📁 检查类目录: {class_name}")
        print(f"   完整路径: {class_dir}")

        # 使用 os.path.exists 检查目录是否存在
        if not os.path.exists(class_dir):
            print(f"   ⚠️  警告：目录不存在，跳过")
            continue

        if not os.path.isdir(class_dir):
            print(f"   ⚠️  警告：路径存在但不是目录，跳过")
            continue

        found_classes.append(class_name)

        # 获取该类目录下的所有文件
        try:
            all_files = os.listdir(class_dir)
            print(f"   ✓ 目录有效，包含 {len(all_files)} 个文件/文件夹")
        except Exception as e:
            print(f"   ⚠️  无法读取目录内容: {e}")
            continue

        # 查找图像和NPY文件对
        img_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = [f for f in all_files if f.lower().endswith(img_extensions)]
        npy_files = [f for f in all_files if f.lower().endswith('.npy')]

        print(f"   - 图像文件数: {len(image_files)}")
        print(f"   - NPY文件数: {len(npy_files)}")

        # 匹配图像和NPY文件对
        matched_count = 0
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            npy_file = f"{base_name}.npy"

            img_path = os.path.join(class_dir, img_file)
            npy_path = os.path.join(class_dir, npy_file)

            if os.path.exists(npy_path):
                file_pairs.append((img_path, npy_path))
                labels_text.append(class_name)
                matched_count += 1
            else:
                # 仅在第一个匹配失败时打印详细信息
                if matched_count == 0:
                    print(f"   ⚠️  示例：找到图像 {img_file} 但缺少对应的 NPY 文件")

        print(f"   ✓ 成功匹配 {matched_count} 对 image-npy 文件")

    # 总结
    print(f"\n{'=' * 80}")
    print(f"📊 数据加载汇总:")
    print(f"   - 期望类别数: {len(classes)}")
    print(f"   - 找到的类别: {len(found_classes)} → {found_classes}")
    print(f"   - 总样本数: {len(file_pairs)}")
    if file_pairs:
        print(f"   - 类别分布: {Counter(labels_text)}")
    print(f"{'=' * 80}\n")

    if not file_pairs:
        # 提供详细的错误信息
        error_msg = f"❌ 在 {data_dir} 中未找到任何有效的 image-npy 文件对。\n"
        error_msg += f"   请检查:\n"
        error_msg += f"   1. 目录结构是否正确（每个类别应有单独的子目录）\n"
        error_msg += f"   2. 每个图像文件是否有对应的同名 .npy 文件\n"
        error_msg += f"   3. 文件扩展名是否正确（支持: .png, .jpg, .jpeg, .bmp）\n"
        if all_subdirs:
            error_msg += f"   4. 实际子目录: {all_subdirs}\n"
            error_msg += f"   5. 期望子目录: {classes}\n"
        raise ValueError(error_msg)

    return file_pairs, labels_text


def test_model(backbone_name):
    """
    Evaluate the specified model on the test set and return detailed metrics
    """
    print(f"\n{'#' * 100}")
    print(f"{'#' * 100}")
    print(f"正在评估模型: {backbone_name}")
    print(f"{'#' * 100}")
    print(f"{'#' * 100}")

    model_path = Config.best_model_paths.get(backbone_name)
    if not model_path or not os.path.exists(model_path):
        print(f"❌ 错误：模型权重文件不存在: {model_path}")
        print(f"   跳过该模型的评估。")
        return None

    try:
        test_pairs, test_labels_text = load_data_from_folder(Config.test_data_root, Config.classes)
    except ValueError as e:
        print(f"❌ 加载测试数据失败:\n{e}")
        print(f"   跳过该模型的评估。")
        return None
    except Exception as e:
        print(f"❌ 加载测试数据时发生未知错误: {e}")
        print(f"   跳过该模型的评估。")
        return None

    label_encoder = LabelEncoder()
    label_encoder.fit(Config.classes)
    test_labels = label_encoder.transform(test_labels_text)

    print(f"✓ 测试数据加载成功")
    print(f"  - 测试样本数: {len(test_labels)}")
    print(f"  - 类别分布: {Counter(test_labels_text)}")

    _, test_transform = get_transforms()
    npy_test_transform = transforms.Compose([
        transforms.Resize(Config.image_size, antialias=True)
    ])

    test_data = [(p[0], p[1], lbl) for p, lbl in zip(test_pairs, test_labels)]
    test_dataset = MultimodalDataset(test_data, transform=test_transform, npy_transform=npy_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, num_workers=4, pin_memory=True)

    dummy_local_path = Config.local_pretrained_paths.get(backbone_name)
    model = ImprovedMultimodalModel(
        num_classes=len(Config.classes),
        backbone_name=backbone_name,
        local_pretrained_path=dummy_local_path
    )

    try:
        model.load_state_dict(torch.load(model_path, map_location=Config.device), strict=True)
        print(f"✓ 成功加载模型权重: {model_path}")
    except RuntimeError as e:
        print(f"❌ 加载模型权重失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 加载模型权重时发生未知错误: {e}")
        return None

    model.to(Config.device)
    model.eval()

    all_preds = []
    all_labels = []

    print(f"开始模型推理...")
    with torch.no_grad():
        for batch_idx, ((img, npy), labels) in enumerate(test_loader):
            img, npy, labels = img.to(Config.device), npy.to(Config.device), labels.to(Config.device)
            outputs = model((img, npy))
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 {(batch_idx + 1) * Config.batch_size} / {len(test_dataset)} 个样本")

    # 计算总体指标
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }

    print(f"\n✓ 模型评估完成:")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")

    return metrics


def plot_training_metrics(log_files, metric_info, save_dir):
    """
    Plot training metrics curves with English labels
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8), dpi=300)

    main_metric_label = metric_info['main_label']
    save_filename = metric_info['filename']
    plot_keys = metric_info['keys']
    save_path = os.path.join(save_dir, save_filename)

    # 使用默认字体，确保英文正常显示
    plt.title(f'{main_metric_label} Curves for All Models', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(main_metric_label, fontsize=14)

    plotted_models = 0
    for model_name, file_path in log_files.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                epochs = df['epoch']
                for key, label in plot_keys.items():
                    if key in df.columns:
                        metric_values = df[key]
                        plt.plot(epochs, metric_values,
                                 label=f"{model_name} - {label}",
                                 linewidth=2)
                        plotted_models += 1
            except KeyError as e:
                print(f"⚠️  警告：读取 {model_name} 的日志文件时缺少必需的列: {e}")
                continue
            except Exception as e:
                print(f"⚠️  警告：读取 {model_name} 的日志文件时发生错误: {e}")
                continue
        else:
            print(f"⚠️  警告：日志文件不存在: {file_path}")

    if plotted_models > 0:
        plt.legend(fontsize=12)
        plt.tight_layout()

        with PdfPages(save_path) as pdf:
            pdf.savefig()
        plt.close()
        print(f"✓ {main_metric_label} 曲线已保存至: {save_path}")
    else:
        plt.close()
        print(f"⚠️  警告：没有可绘制的数据，跳过 {main_metric_label} 曲线生成")


if __name__ == "__main__":
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)

    models_to_test = [
        'resnet34',
        'resnet50',
        'efficientnet_b0',
        'vit_base_patch16_224',
        'swin_tiny_patch4_window7_224',
        'convnext_tiny'
    ]

    log_files = {
        'ConvNeXt_Tiny': './results/result6/convnext_tiny/training_log_convnext_tiny.csv',
        'EfficientNet_B0': './results/result6/efficientnet_b0/training_log_efficientnet_b0.csv',
        'ResNet34': './results/result6/ResNet34/training_log_resnet34.csv',
        'ResNet50': './results/result6/ResNet50/training_log_resnet50.csv',
        'ViT_Base_Patch16_224': './results/result6/vit_base_patch16_224/training_log_vit_base_patch16_224.csv'
    }

    test_results = {}

    print("\n" + "=" * 100)
    print("开始模型评估流程")
    print("=" * 100)

    for model_name in models_to_test:
        if ('vit' in model_name or 'swin' in model_name or
                'efficientnet' in model_name or 'convnext' in model_name):
            if timm is None:
                print(f"⚠️  跳过 {model_name}：未安装 timm 库")
                continue
        try:
            metrics = test_model(model_name)
            if metrics:
                test_results[model_name] = metrics
            else:
                test_results[model_name] = "Skipped"
        except Exception as e:
            print(f"❌ 测试 {model_name} 时发生未知错误: {e}")
            import traceback

            traceback.print_exc()
            test_results[model_name] = "Error"

    print("\n" + "=" * 100)
    print("=" * 100)
    print("最终测试结果汇总")
    print("=" * 100)
    print("=" * 100)

    for model_name, metrics in test_results.items():
        if isinstance(metrics, dict):
            print(f"\n✓ 模型: {model_name}")
            print(f"    Accuracy:  {metrics['Accuracy']:.4f}")
            print(f"    Precision: {metrics['Precision']:.4f}")
            print(f"    Recall:    {metrics['Recall']:.4f}")
            print(f"    F1-Score:  {metrics['F1-Score']:.4f}")
        else:
            print(f"\n✗ 模型: {model_name} - 评估失败 ({metrics})")

    print("\n" + "=" * 100)
    print("开始绘制训练指标曲线")
    print("=" * 100)

    # 绘制并保存训练准确率曲线图 (英文)
    accuracy_metrics_to_plot = {
        'main_label': 'Training Accuracy',
        'filename': 'training_accuracy.pdf',
        'keys': {
            'train_accuracy': 'Training Accuracy',
        }
    }
    plot_training_metrics(log_files, accuracy_metrics_to_plot, Config.results_save_dir)

    # 绘制并保存训练损失曲线图 (英文)
    loss_metrics_to_plot = {
        'main_label': 'Training Loss',
        'filename': 'training_loss.pdf',
        'keys': {
            'train_loss': 'Training Loss',
        }
    }
    plot_training_metrics(log_files, loss_metrics_to_plot, Config.results_save_dir)

    print("\n" + "=" * 100)
    print("✓ 所有任务完成！")
    print("=" * 100)