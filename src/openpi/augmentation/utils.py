"""
Utility functions for augmentation pipeline creation and management.
"""

from typing import List, Dict, Any, Optional
from openpi.transforms import DataTransformFn
from .image_transforms import ImageAugmentation, ColorJitterTransform


def create_augmentation_pipeline(
    enable_image_aug: bool = True,
    image_aug_params: Optional[Dict[str, Any]] = None,
    enable_color_jitter: bool = False,
    color_jitter_params: Optional[Dict[str, Any]] = None,
) -> List[DataTransformFn]:
    """
    Create a list of augmentation transforms based on configuration.
    
    Args:
        enable_image_aug: Whether to include comprehensive image augmentation
        image_aug_params: Parameters for ImageAugmentation transform
        enable_color_jitter: Whether to include color jittering (alternative to image_aug)
        color_jitter_params: Parameters for ColorJitterTransform
    
    Returns:
        List of augmentation transforms to be applied in order
    """
    transforms = []
    
    if enable_image_aug:
        if image_aug_params is None:
            # Default parameters matching your requirements
            image_aug_params = {
                "random_resized_crop": {"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
                "random_brightness": [0.2],
                "random_contrast": [0.8, 1.2],
                "random_saturation": [0.8, 1.2],
                "random_hue": [0.05],
                "augment_order": [
                    "random_resized_crop", "random_brightness", "random_contrast", 
                    "random_saturation", "random_hue"
                ],
                "augment_prob": 0.8
            }
        transforms.append(ImageAugmentation(**image_aug_params))
    
    elif enable_color_jitter:
        if color_jitter_params is None:
            color_jitter_params = {
                "brightness": 0.2,
                "contrast": (0.8, 1.2),
                "saturation": (0.8, 1.2), 
                "hue": 0.05,
                "prob": 0.8
            }
        transforms.append(ColorJitterTransform(**color_jitter_params))
    
    return transforms


def get_default_libero_augmentation() -> ImageAugmentation:
    """
    Get default augmentation configuration optimized for Libero tasks.
    
    Returns:
        ImageAugmentation transform with Libero-optimized parameters
    """
    return ImageAugmentation(
        random_resized_crop={"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
        random_brightness=[0.2],
        random_contrast=[0.8, 1.2],
        random_saturation=[0.8, 1.2],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop", "random_brightness", "random_contrast", 
            "random_saturation", "random_hue"
        ],
        augment_prob=0.8
    )


def get_conservative_augmentation() -> ColorJitterTransform:
    """
    Get conservative augmentation for sensitive training scenarios.
    
    Returns:
        ColorJitterTransform with mild augmentation parameters
    """
    return ColorJitterTransform(
        brightness=0.1,
        contrast=(0.9, 1.1),
        saturation=(0.9, 1.1),
        hue=0.02,
        prob=0.5
    )


def get_aggressive_augmentation() -> ImageAugmentation:
    """
    Get aggressive augmentation for robust training scenarios.
    
    Returns:
        ImageAugmentation with stronger augmentation parameters
    """
    return ImageAugmentation(
        random_resized_crop={"scale": [0.8, 1.0], "ratio": [0.9, 1.1]},
        random_brightness=[0.3],
        random_contrast=[0.7, 1.3],
        random_saturation=[0.7, 1.3],
        random_hue=[0.1],
        augment_order=[
            "random_resized_crop", "random_brightness", "random_contrast", 
            "random_saturation", "random_hue"
        ],
        augment_prob=0.9
    )
