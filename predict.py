"""
烟草高光谱图像分类 - 单样本预测脚本
支持命令行和交互式两种使用方式
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import argparse

# 尝试导入 timm
try:
    import timm

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("⚠️  警告: timm 库未安装，部分模型不可用（EfficientNet, ViT, ConvNeXt）")
    print("   可运行: pip install timm\n")


# ==================== 配置类 ====================
class Config:
    """全局配置"""
    classes = ["微带青", "杂色", "柠檬黄", "橘黄", "红棕", "青色"]
    image_size = (224, 224)
    max_npy_channels = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = 512

    # 模型权重路径
    model_weights = {
        'resnet34': './results/result1/ResNet34/best_model_resnet34.pth',
        'resnet50': './results/result1/ResNet50/best_model_resnet50.pth',
        'efficientnet_b0': './results/result1/efficientnet_b0/best_model_efficientnet_b0.pth',
        'vit_base_patch16_224': './results/result1/vit_base_patch16_224/best_model_vit_base_patch16_224.pth',
        'convnext_tiny': './results/result1/convnext_tiny/best_model_convnext_tiny.pth',
    }

    # 预训练模型路径（仅用于模型结构初始化）
    pretrained_paths = {
        'resnet34': "./pre_models/resnet34.pth",
        'resnet50': "./pre_models/resnet50.pth",
        'efficientnet_b0': "./pre_models/efficientnet_b0.bin",
        'vit_base_patch16_224': "./pre_models/vit_base_patch16_224.bin",
        'convnext_tiny': "./pre_models/convnext_tiny.bin",
    }


# ==================== 模型定义 ====================
class MultimodalModel(nn.Module):
    """多模态融合模型（图像 + 高光谱）"""

    def __init__(self, num_classes, backbone_name='resnet34'):
        super().__init__()
        self.backbone_name = backbone_name

        # 图像特征提取器
        self.img_backbone, self.feature_dim_img = self._build_image_backbone(backbone_name)

        # 高光谱特征提取器（名称改为 npy_feature_extractor，与训练时一致）
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

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim_img + Config.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _build_image_backbone(self, backbone_name):
        """构建图像骨干网络"""
        name = backbone_name.lower()

        if name == 'resnet34':
            model = models.resnet34(weights=None)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()

        elif name == 'resnet50':
            model = models.resnet50(weights=None)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()

        elif name == 'efficientnet_b0':
            if not HAS_TIMM:
                raise ImportError("EfficientNet 需要 timm 库，请运行: pip install timm")
            model = timm.create_model('efficientnet_b0', pretrained=False)
            feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()

        elif name == 'vit_base_patch16_224':
            if not HAS_TIMM:
                raise ImportError("ViT 需要 timm 库，请运行: pip install timm")
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            feature_dim = model.head.in_features
            model.head = nn.Identity()

        elif name == 'convnext_tiny':
            if not HAS_TIMM:
                raise ImportError("ConvNeXt 需要 timm 库，请运行: pip install timm")
            model = timm.create_model('convnext_tiny', pretrained=False)
            feature_dim = model.head.fc.in_features
            model.head.fc = nn.Identity()

        else:
            raise ValueError(f"不支持的模型: {backbone_name}")

        return model, feature_dim

    def forward(self, x):
        img, npy = x

        # 提取图像特征
        img_feat = self.img_backbone(img)
        if img_feat.ndim == 3 and ('vit' in self.backbone_name or 'swin' in self.backbone_name):
            img_feat = img_feat[:, 0]  # 取 [CLS] token
        elif img_feat.ndim > 2:
            img_feat = torch.flatten(img_feat, 1)

        # 提取高光谱特征
        npy_feat = self.npy_feature_extractor(npy)
        npy_feat = torch.flatten(npy_feat, 1)

        # 特征融合 + 分类
        combined = torch.cat([img_feat, npy_feat], dim=1)
        return self.classifier(combined)


# ==================== 数据预处理 ====================
def load_and_preprocess_image(img_path):
    """加载并预处理图像"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图像文件不存在: {img_path}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    return img_tensor


def load_and_preprocess_npy(npy_path):
    """加载并预处理高光谱数据"""
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"NPY 文件不存在: {npy_path}")

    # 加载数据
    npy_data = np.load(npy_path).astype(np.float32)

    # 确保是 3D (H, W, C)
    if npy_data.ndim == 2:
        npy_data = np.expand_dims(npy_data, axis=-1)

    # 通道数对齐
    current_channels = npy_data.shape[-1]
    if current_channels < Config.max_npy_channels:
        # 零填充
        padded = np.zeros((*npy_data.shape[:2], Config.max_npy_channels), dtype=np.float32)
        padded[:, :, :current_channels] = npy_data
        npy_data = padded
    elif current_channels > Config.max_npy_channels:
        # 截断
        npy_data = npy_data[:, :, :Config.max_npy_channels]

    # Min-Max 归一化（使用百分位数避免异常值）
    min_val = np.percentile(npy_data, 5)
    max_val = np.percentile(npy_data, 95)
    npy_range = max_val - min_val

    if npy_range < 1e-6:
        npy_data = np.full_like(npy_data, 0.5)
    else:
        npy_data = (npy_data - min_val) / npy_range
    npy_data = np.clip(npy_data, 0, 1)

    # 转换为 Tensor (C, H, W)
    npy_tensor = torch.from_numpy(npy_data).permute(2, 0, 1).float()

    # Resize
    resize_transform = transforms.Resize(Config.image_size, antialias=True)
    npy_tensor = resize_transform(npy_tensor).unsqueeze(0)  # (1, 6, 224, 224)

    return npy_tensor


# ==================== 模型加载 ====================
def load_model(model_name):
    """加载训练好的模型"""
    # 检查模型权重是否存在
    weight_path = Config.model_weights.get(model_name)
    if not weight_path:
        raise ValueError(f"未知的模型名称: {model_name}\n可用模型: {list(Config.model_weights.keys())}")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"模型权重文件不存在: {weight_path}\n"
            f"请确保已训练该模型并保存权重文件。"
        )

    # 初始化模型结构
    print(f"⏳ 正在加载模型: {model_name}...")
    model = MultimodalModel(
        num_classes=len(Config.classes),
        backbone_name=model_name
    )

    # 加载权重
    try:
        state_dict = torch.load(weight_path, map_location=Config.device)
        model.load_state_dict(state_dict)
        model.to(Config.device)
        model.eval()
        print(f"✓ 模型加载成功")
        return model
    except Exception as e:
        raise RuntimeError(f"加载模型权重失败: {e}")


# ==================== 预测函数 ====================
def predict(img_path, npy_path, model_name='resnet34', show_all_probs=True):
    """
    执行单样本预测

    Args:
        img_path: 图像文件路径
        npy_path: NPY 文件路径
        model_name: 模型名称
        show_all_probs: 是否显示所有类别的概率

    Returns:
        dict: 预测结果字典
    """
    # 打印标题
    print(f"\n{'=' * 100}")
    print(f"🌿 烟草颜色分类预测系统")
    print(f"{'=' * 100}")
    print(f"📷 图像文件: {os.path.basename(img_path)}")
    print(f"📊 数据文件: {os.path.basename(npy_path)}")
    print(f"🤖 使用模型: {model_name}")
    print(f"💻 运行设备: {Config.device}")
    print(f"{'=' * 100}\n")

    # 1. 加载数据
    print("📥 正在加载数据...")
    try:
        img_tensor = load_and_preprocess_image(img_path)
        npy_tensor = load_and_preprocess_npy(npy_path)
        print(f"  ✓ 图像尺寸: {img_tensor.shape}")
        print(f"  ✓ NPY尺寸:  {npy_tensor.shape}")
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        return None

    # 2. 加载模型
    try:
        model = load_model(model_name)
    except Exception as e:
        print(f"  ❌ {e}")
        return None

    # 3. 执行推理
    print(f"\n🔮 正在预测...")
    with torch.no_grad():
        img_tensor = img_tensor.to(Config.device)
        npy_tensor = npy_tensor.to(Config.device)

        outputs = model((img_tensor, npy_tensor))
        probabilities = torch.softmax(outputs, dim=1)[0]  # (num_classes,)

        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

    predicted_class = Config.classes[predicted_idx]

    # 4. 输出结果
    print(f"\n{'=' * 100}")
    print(f"✅ 预测完成")
    print(f"{'=' * 100}")
    print(f"🎯 预测类别: \033[1;32m{predicted_class}\033[0m")
    print(f"📈 置信度:   \033[1;36m{confidence:.2%}\033[0m")

    # 显示所有类别概率
    if show_all_probs:
        print(f"\n{'─' * 100}")
        print(f"📊 所有类别的概率分布:")
        print(f"{'─' * 100}")

        # 按概率排序
        probs_with_class = [(Config.classes[i], probabilities[i].item())
                            for i in range(len(Config.classes))]
        probs_with_class.sort(key=lambda x: x[1], reverse=True)

        for rank, (cls, prob) in enumerate(probs_with_class, 1):
            # 生成进度条
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)

            # 高亮显示预测类别
            if cls == predicted_class:
                print(f"  {rank}. \033[1;32m{cls:10s}\033[0m │{bar}│ {prob:6.2%}")
            else:
                print(f"  {rank}. {cls:10s} │{bar}│ {prob:6.2%}")

        print(f"{'─' * 100}")

    print(f"{'=' * 100}\n")

    # 返回结果字典
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': {Config.classes[i]: probabilities[i].item()
                          for i in range(len(Config.classes))},
        'model_name': model_name
    }

    return result


# ==================== 交互式输入 ====================
def interactive_mode():
    """交互式输入模式"""
    print("\n" + "=" * 100)
    print("🌿 烟草颜色分类预测系统 - 交互式模式")
    print("=" * 100)

    # 输入文件路径
    print("\n请输入文件路径（可直接拖拽文件到终端）:")
    img_path = input("📷 图像文件路径: ").strip().strip('"').strip("'")
    npy_path = input("📊 NPY文件路径:  ").strip().strip('"').strip("'")

    # 选择模型
    print("\n可用模型:")
    available_models = list(Config.model_weights.keys())
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")

    model_choice = input(f"\n选择模型 (1-{len(available_models)}) [默认: 1]: ").strip()
    if not model_choice:
        model_choice = '1'

    try:
        model_idx = int(model_choice) - 1
        model_name = available_models[model_idx]
    except (ValueError, IndexError):
        print("⚠️  无效选择，使用默认模型: resnet34")
        model_name = 'resnet34'

    # 执行预测
    predict(img_path, npy_path, model_name, show_all_probs=True)


# ==================== 命令行接口 ====================
def main():
    parser = argparse.ArgumentParser(
        description='烟草颜色分类 - 单样本预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 命令行模式
  python predict.py --img test.jpg --npy test.npy --model resnet34

  # 交互式模式
  python predict.py

  # Windows 路径示例
  python predict.py --img "F:\\data\\test\\微带青\\sample.jpg" --npy "F:\\data\\test\\微带青\\sample.npy"

可用模型:
  - resnet34 (默认)
  - resnet50
  - efficientnet_b0 (需要 timm)
  - vit_base_patch16_224 (需要 timm)
  - convnext_tiny (需要 timm)
        """
    )

    parser.add_argument('--img', type=str, help='图像文件路径')
    parser.add_argument('--npy', type=str, help='NPY 文件路径')
    parser.add_argument('--model', type=str, default='resnet34',
                        choices=list(Config.model_weights.keys()),
                        help='模型名称 (默认: resnet34)')
    parser.add_argument('--hide-probs', action='store_true',
                        help='不显示所有类别的概率分布')

    args = parser.parse_args()

    # 判断运行模式
    if args.img and args.npy:
        # 命令行模式
        predict(
            img_path=args.img,
            npy_path=args.npy,
            model_name=args.model,
            show_all_probs=not args.hide_probs
        )
    else:
        # 交互式模式
        if args.img or args.npy:
            print("⚠️  警告: 需要同时提供 --img 和 --npy 参数")
            print("   切换到交互式模式...\n")
        interactive_mode()


if __name__ == "__main__":
    main()