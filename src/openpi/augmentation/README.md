# Data Augmentation Usage Guide

This guide shows how to use the new data augmentation capabilities in OpenPI.

## Quick Start

### 1. Using Pre-configured Augmentation

```python
from openpi.training.config import LeRobotLiberoDataConfigWithAug

# Create augmented training config
config = TrainConfig(
    name="my_augmented_training",
    model=pi0.Pi0Config(action_dim=7, action_horizon=10),
    data=LeRobotLiberoDataConfigWithAug(
        repo_id="your/dataset",
        enable_image_aug=True,  # Enable augmentation
        aug_prob=0.8,          # 80% chance to apply augmentation
    )
)
```

### 2. Custom Augmentation Parameters

```python
config = LeRobotLiberoDataConfigWithAug(
    repo_id="your/dataset",
    enable_image_aug=True,
    
    # Probability settings
    aug_prob=0.8,
    
    # Crop settings
    crop_scale=(0.9, 0.9),      # Scale range for random crop
    crop_ratio=(1.0, 1.0),      # Aspect ratio range
    
    # Color settings
    brightness_range=0.2,        # ±0.2 brightness variation
    contrast_range=(0.8, 1.2),   # 0.8x to 1.2x contrast
    saturation_range=(0.8, 1.2), # 0.8x to 1.2x saturation
    hue_range=0.05,              # ±0.05 hue shift
    
    # Execution order
    aug_order=("random_resized_crop", "random_brightness", 
               "random_contrast", "random_saturation", "random_hue")
)
```

### 3. Using Direct Transform

```python
from openpi.augmentation import ImageAugmentation

# Create transform directly
aug_transform = ImageAugmentation(
    random_resized_crop={"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
    random_brightness=[0.2],
    random_contrast=[0.8, 1.2],
    random_saturation=[0.8, 1.2],
    random_hue=[0.05],
    augment_prob=0.8
)

# Add to existing data config
enhanced_transforms = base_config.data_transforms.push(
    inputs=[aug_transform]
)
```

### 4. Preset Configurations

```python
from openpi.augmentation.utils import (
    get_default_libero_augmentation,
    get_conservative_augmentation,
    get_aggressive_augmentation
)

# Use presets
conservative_aug = get_conservative_augmentation()  # Mild augmentation
default_aug = get_default_libero_augmentation()     # Balanced for Libero
aggressive_aug = get_aggressive_augmentation()      # Strong augmentation
```

## Augmentation Pipeline

The augmentation transforms are applied in this order within the data pipeline:

```
Raw Data → RepackTransforms → [IMAGE AUGMENTATION] → LiberoInputs → DeltaActions → Normalize → ModelTransforms → Model
```

## Training Examples

### Conservative Training (Sensitive Tasks)
```python
TrainConfig(
    name="conservative_aug_training",
    data=LeRobotLiberoDataConfigWithAug(
        enable_image_aug=True,
        aug_prob=0.5,           # Lower probability
        brightness_range=0.1,   # Mild brightness
        contrast_range=(0.9, 1.1),  # Mild contrast
    )
)
```

### Robust Training (General Tasks)
```python
TrainConfig(
    name="robust_aug_training", 
    data=LeRobotLiberoDataConfigWithAug(
        enable_image_aug=True,
        aug_prob=0.8,           # Standard probability
        brightness_range=0.2,   # Standard brightness
        contrast_range=(0.8, 1.2),  # Standard contrast
    )
)
```

### Aggressive Training (Challenging Scenarios)
```python
TrainConfig(
    name="aggressive_aug_training",
    data=LeRobotLiberoDataConfigWithAug(
        enable_image_aug=True,
        aug_prob=0.9,           # High probability
        brightness_range=0.3,   # Strong brightness
        contrast_range=(0.7, 1.3),  # Strong contrast
        crop_scale=(0.8, 1.0),  # More aggressive cropping
    )
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the augmentation module is properly installed
2. **Performance**: Start with lower augmentation probability (0.5) and increase gradually
3. **Overfitting**: If model performance decreases, reduce augmentation strength
4. **Memory Usage**: Image augmentation increases memory usage during training

### Best Practices

1. **Start Conservative**: Begin with mild augmentation and increase gradually
2. **Monitor Performance**: Track validation metrics to ensure augmentation helps
3. **Task-Specific Tuning**: Different tasks may benefit from different augmentation types
4. **Balance Speed vs. Quality**: More augmentation improves robustness but slows training

## API Reference

See the individual module documentation:
- `openpi.augmentation.ImageAugmentation`: Main augmentation transform
- `openpi.augmentation.ColorJitterTransform`: Lightweight color-only augmentation
- `openpi.augmentation.utils`: Utility functions and presets
