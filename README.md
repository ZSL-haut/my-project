
# Tobacco Hyperspectral Image Classification Project

## 📋 Project Overview

This project utilizes hyperspectral image data to automatically classify multiple quality attributes of tobacco, including six dimensions: color, maturity, structure, body, oil content, and chroma.

## 🏷️ Classification Categories

### 1. Color Classification (6 classes)

```python
classes = ["Slightly Green", "Mixed Color", "Lemon Yellow", "Orange Yellow", "Reddish Brown", "Green"]
```

### 2. Maturity Classification (5 classes)

```python
classes = ["Mature", "Fully Mature", "Moderately Mature", "Undermature", "False Mature"]
```

### 3. Structure Classification (4 classes)

```python
classes = ["Loose", "Moderately Loose", "Slightly Dense", "Dense"]
```

### 4. Body Classification (3 classes)

```python
classes = ["Medium", "Thick", "Thin"]
```

### 5. Oil Content Classification (4 classes)

```python
classes = ["Abundant", "Present", "Slightly Present", "Low"]
```

### 6. Chroma Classification (5 classes)

```python
classes = ["Strong", "Intense", "Medium", "Weak", "Light"]
```

## 🚀 Usage

### Environment Setup

#### Install Dependencies

```bash
pip install torch torchvision timm numpy pandas scikit-learn pillow matplotlib
```

### Model Training

Execute in terminal:

```bash
python train.py
```

### Model Evaluation

Execute in terminal:

```bash
python evaluate.py
```


