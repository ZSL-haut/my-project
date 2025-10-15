烟草高光谱图像分类项目
📋 项目简介
本项目基于高光谱图像数据，对烟草的多个质量属性进行自动分类，包括颜色、成熟度、结构、身份、油分和色度等六个维度。

📊 数据集说明
样本采集
采集地点：河南省 48 个烟草站
样本数量：378 个高光谱 HDR 文件
地域代表性：广泛覆盖不同产区
采集环境
为确保数据质量，所有样本均在标准化实验环境下采集：

表格
环境参数	标准范围
色温	5300–5800 K
工作面光照度	1800–2200 Lx
温度	20–24 ℃
相对湿度	65–75 %
说明：标准化环境可排除环境光照、温湿度等因素对数据采集的干扰。

数据格式
原始数据尺寸：(856, 1092, 480) 或 (857, 1092, 480)
空间维度：856/857 × 1092 像素
光谱维度：480 个波段
文件格式：HDR（高光谱数据格式）
数据标注
由烟草专家对以下六个质量属性进行评分和分类：

颜色 - 反映烟叶外观色泽
成熟度 - 评估烟叶成熟程度
结构 - 描述烟叶组织结构
身份 - 评价烟叶厚薄程度
油分 - 衡量烟叶含油量
色度 - 评估颜色饱和度
数据增强
原因：原始数据存在类别不平衡问题
方法：随机翻转、亮度调整等
结果：每个类别扩充至 500 个样本
🏷️ 分类类别
1. 颜色分类（6 类）
python
classes = ["微带青", "杂色", "柠檬黄", "橘黄", "红棕", "青色"]
2. 成熟度分类（5 类）
python
classes = ["成熟", "完熟", "尚熟", "欠熟", "假熟"]
3. 结构分类（4 类）
python
classes = ["疏松", "尚疏松", "稍密", "紧密"]
4. 身份分类（3 类）
python
classes = ["中等", "厚", "薄"]
5. 油分分类（4 类）
python
classes = ["多", "有", "稍有", "少"]
6. 色度分类（5 类）
python
classes = ["浓", "强", "中", "弱", "淡"]
🚀 使用方法
环境准备
1. 安装依赖
bash
pip install torch torchvision timm numpy pandas scikit-learn pillow matplotlib
2. 目录结构
python
项目根目录/
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
├── datas1/               # 数据集目录
│   ├── train/            # 训练集
│   │   ├── 微带青/
│   │   ├── 杂色/
│   │   ├── 柠檬黄/
│   │   └── ...
│   └── val/              # 验证集
│       ├── 微带青/
│       └── ...
├── pre_models/           # 预训练权重目录
│   ├── resnet34.pth
│   ├── resnet50.pth
│   ├── efficientnet_b0.pth
│   ├── vit_base_patch16_224.bin
│   └── convnext_tiny.bin
└── results/              # 结果输出目录（自动创建）
    └── result1/
        ├── resnet34/
        ├── resnet50/
        └── ...
模型训练
在终端中执行：

bash
python train.py
输出结果：

训练日志：results/result1/<model_name>/training_log_<model_name>.csv
最佳模型：results/result1/<model_name>/best_model_<model_name>.pth
训练曲线：results/result1/<model_name>/training_metrics_<model_name>.png
模型评估
在终端中执行：

bash
python evaluate.py
输出结果：

评估指标（准确率、精确率、召回率、F1 分数）
混淆矩阵
分类报告
🎯 支持的模型
项目支持以下预训练模型进行对比训练：

表格
模型名称	来源	特点
ResNet-34	torchvision	经典卷积网络，参数量适中
ResNet-50	torchvision	更深的 ResNet 变体，性能更强
EfficientNet-B0	timm	高效的轻量级模型
ViT-Base	timm	Vision Transformer，全局建模能力强
ConvNeXt-Tiny	timm	现代化卷积网络，性能优异
📈 训练配置
超参数设置
python
# 数据相关
image_size = (224, 224)
batch_size = 32
max_npy_channels = 6

# 训练相关
num_epochs = 100
lr_pretrained = 5e-5      # 预训练层学习率
lr_new = 1e-3             # 新增层学习率
weight_decay = 1e-4
patience = 20             # 早停轮数
数据增强
训练集增强：

随机裁剪（scale=0.5–1.0）
随机水平翻转
随机垂直翻转
颜色抖动（亮度、对比度、饱和度、色调）
随机旋转（±15°）
验证集增强：

中心裁剪
标准化
📊 数据模态
多模态融合
项目采用 双模态融合 策略：

RGB 图像模态

输入尺寸：224 × 224 × 3
特征提取：预训练 CNN/Transformer
高光谱 NPY 模态

输入尺寸：224 × 224 × 6（通道数）
特征提取：自定义 CNN
融合策略

特征级融合：拼接两个模态的特征向量
联合分类：全连接层进行最终分类
🔧 高级配置
修改分类任务
在 train.py 或 evaluate.py 中修改 Config.classes：

python
# 示例：切换到成熟度分类任务
class Config:
    classes = ["成熟", "完熟", "尚熟", "欠熟", "假熟"]
    # ... 其他配置
修改数据路径
python
class Config:
    train_data_root = './datas1/train'      # 训练集路径
    val_data_root = './datas1/val'          # 验证集路径
    output_dir = './results/result1'        # 输出路径
修改模型列表
python
# 在 train.py 的 main 函数中修改
models_to_compare = [
    'resnet34',
    'resnet50',
    'efficientnet_b0',
    'vit_base_patch16_224',
    'convnext_tiny'
]
📝 输出文件说明
训练输出
每个模型训练完成后，会在 results/result1/<model_name>/ 生成：

best_model_<model_name>.pth

验证集准确率最高的模型权重
training_log_<model_name>.csv

每个 epoch 的训练/验证损失和准确率
学习率变化记录
training_metrics_<model_name>.png

损失曲线图
准确率曲线图
评估输出
评估完成后输出：

整体准确率
每个类别的精确率、召回率、F1 分数
混淆矩阵
分类错误样本分析
⚠️ 注意事项
1. 硬件要求
GPU 内存：建议 ≥ 8 GB（用于大型模型如 ViT）
RAM：建议 ≥ 16 GB
2. 环境兼容性
Python 版本：3.8+
PyTorch 版本：1.12+
CUDA 版本：根据 PyTorch 版本匹配
3. 数据格式要求
每个类别文件夹内需包含配对的 图像文件（.png/.jpg）和 NPY 文件（.npy）
文件命名必须一致（例如：sample_001.png 和 sample_001.npy）
4. 预训练权重
需要预先下载对应模型的预训练权重
放置在 pre_models/ 目录下
权重文件名需与配置中的路径一致
📧 联系方式
如有问题或建议，请联系项目维护者。

📄 许可证
本项目仅供学术研究使用。


