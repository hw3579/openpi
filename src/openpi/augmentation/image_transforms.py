"""
Image augmentation transforms for OpenPI framework.

This module implements various image augmentation techniques to improve model robustness
and generalization. All transforms follow the OpenPI DataTransformFn protocol.
"""

import dataclasses
from typing import Dict, List, Tuple, Optional, Union, Sequence
import jax
import jax.numpy as jnp
import numpy as np
from openpi.transforms import DataTransformFn, DataDict


@dataclasses.dataclass(frozen=True)
class ImageAugmentation(DataTransformFn):
    """
    Comprehensive image augmentation transform for robotics data.
    
    Supports multiple augmentation types that are commonly effective for robotic vision:
    - Random resized crop: Simulates different viewpoints and scales
    - Color jittering: Brightness, contrast, saturation, hue adjustments
    - Configurable execution order and probability
    
    Args:
        random_resized_crop: Dict with 'scale' and 'ratio' parameters for cropping
        random_brightness: List with single value for brightness variation range
        random_contrast: List with [min, max] values for contrast adjustment
        random_saturation: List with [min, max] values for saturation adjustment  
        random_hue: List with single value for hue variation range
        augment_order: Sequence of augmentation names to apply in order
        augment_prob: Probability of applying augmentations (0.0 to 1.0)
        seed: Optional fixed seed for reproducible augmentations
    """
    
    # Random resized crop parameters
    random_resized_crop: Optional[Dict[str, Union[List[float], Tuple[float, float]]]] = None
    # Brightness adjustment range
    random_brightness: Optional[List[float]] = None
    # Contrast adjustment range  
    random_contrast: Optional[List[float]] = None
    # Saturation adjustment range
    random_saturation: Optional[List[float]] = None
    # Hue adjustment range
    random_hue: Optional[List[float]] = None
    # Augmentation execution order
    augment_order: Sequence[str] = dataclasses.field(default_factory=lambda: [
        "random_resized_crop", "random_brightness", "random_contrast", 
        "random_saturation", "random_hue"
    ])
    # Probability of applying augmentations
    augment_prob: float = 0.8
    # Random seed for reproducibility
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate augmentation parameters."""
        if self.random_brightness and len(self.random_brightness) != 1:
            raise ValueError("random_brightness should have exactly 1 element")
        if self.random_contrast and len(self.random_contrast) != 2:
            raise ValueError("random_contrast should have exactly 2 elements")
        if self.random_saturation and len(self.random_saturation) != 2:
            raise ValueError("random_saturation should have exactly 2 elements") 
        if self.random_hue and len(self.random_hue) != 1:
            raise ValueError("random_hue should have exactly 1 element")
        if not 0.0 <= self.augment_prob <= 1.0:
            raise ValueError("augment_prob must be between 0.0 and 1.0")
    
    def __call__(self, data: DataDict) -> DataDict:
        """Apply image augmentations to the data."""
        if "image" not in data:
            return data
            
        # Generate random seed based on data index if available
        if self.seed is not None:
            rng = jax.random.key(self.seed)
        else:
            # Use hash of data content for deterministic but varied seeds
            seed_val = hash(str(id(data))) % (2**31)
            rng = jax.random.key(seed_val)
        
        # Decide whether to apply augmentations
        rng, aug_rng = jax.random.split(rng)
        if jax.random.uniform(aug_rng) > self.augment_prob:
            return data
            
        # Apply augmentations to each image
        augmented_images = {}
        for img_key, img in data["image"].items():
            rng, img_rng = jax.random.split(rng)
            augmented_images[img_key] = self._apply_augmentations(img, img_rng)
        
        # Create new data dict with augmented images
        augmented_data = dict(data)
        augmented_data["image"] = augmented_images
        return augmented_data
    
    def _apply_augmentations(self, image: np.ndarray, rng: jax.random.PRNGKey) -> np.ndarray:
        """Apply augmentation sequence to a single image."""
        img = np.asarray(image, dtype=np.float32)
        
        # Normalize to [0,1] range if needed
        if img.max() > 1.0:
            img = img / 255.0
            
        # Apply augmentations in specified order
        for aug_name in self.augment_order:
            rng, aug_rng = jax.random.split(rng)
            
            if aug_name == "random_resized_crop" and self.random_resized_crop:
                img = self._random_resized_crop(img, aug_rng)
            elif aug_name == "random_brightness" and self.random_brightness:
                img = self._random_brightness(img, aug_rng)
            elif aug_name == "random_contrast" and self.random_contrast:
                img = self._random_contrast(img, aug_rng)
            elif aug_name == "random_saturation" and self.random_saturation:
                img = self._random_saturation(img, aug_rng)
            elif aug_name == "random_hue" and self.random_hue:
                img = self._random_hue(img, aug_rng)
        
        # Ensure output is in valid range
        img = np.clip(img, 0.0, 1.0)
        return img
    
    def _random_resized_crop(self, image: np.ndarray, rng: jax.random.PRNGKey) -> np.ndarray:
        """Apply random resized crop augmentation."""
        params = self.random_resized_crop
        scale_range = params.get("scale", [0.9, 0.9])
        ratio_range = params.get("ratio", [1.0, 1.0])
        
        h, w = image.shape[:2]
        
        # Sample scale factor
        if len(scale_range) == 2 and scale_range[0] != scale_range[1]:
            rng, scale_rng = jax.random.split(rng)
            scale = jax.random.uniform(scale_rng, minval=scale_range[0], maxval=scale_range[1])
        else:
            scale = scale_range[0]
        
        # Sample aspect ratio
        if len(ratio_range) == 2 and ratio_range[0] != ratio_range[1]:
            rng, ratio_rng = jax.random.split(rng)
            ratio = jax.random.uniform(ratio_rng, minval=ratio_range[0], maxval=ratio_range[1])
        else:
            ratio = ratio_range[0]
        
        # Calculate crop dimensions
        crop_area = int(h * w * scale)
        crop_h = int(np.sqrt(crop_area / ratio))
        crop_w = int(crop_h * ratio)
        
        # Ensure crop dimensions don't exceed image size
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        
        if crop_h < h or crop_w < w:
            # Sample crop position
            rng, pos_rng = jax.random.split(rng)
            top = int(jax.random.randint(pos_rng, shape=(), minval=0, maxval=h - crop_h + 1))
            rng, pos_rng = jax.random.split(rng)
            left = int(jax.random.randint(pos_rng, shape=(), minval=0, maxval=w - crop_w + 1))
            
            # Perform crop
            cropped = image[top:top+crop_h, left:left+crop_w]
            
            # For simplicity, we'll resize back using nearest neighbor
            # In production, you might want to use more sophisticated interpolation
            if cropped.shape[:2] != (h, w):
                # Simple resize by repeating/sampling pixels
                resize_h_ratio = h / crop_h
                resize_w_ratio = w / crop_w
                
                new_img = np.zeros_like(image)
                for i in range(h):
                    for j in range(w):
                        src_i = min(int(i / resize_h_ratio), crop_h - 1)
                        src_j = min(int(j / resize_w_ratio), crop_w - 1)
                        new_img[i, j] = cropped[src_i, src_j]
                return new_img
            else:
                return cropped
        
        return image
    
    def _random_brightness(self, image: np.ndarray, rng: jax.random.PRNGKey) -> np.ndarray:
        """Apply random brightness adjustment."""
        brightness_range = self.random_brightness[0]
        rng, bright_rng = jax.random.split(rng)
        delta = float(jax.random.uniform(bright_rng, minval=-brightness_range, maxval=brightness_range))
        return image + delta
    
    def _random_contrast(self, image: np.ndarray, rng: jax.random.PRNGKey) -> np.ndarray:
        """Apply random contrast adjustment."""
        contrast_range = self.random_contrast
        rng, contrast_rng = jax.random.split(rng)
        factor = float(jax.random.uniform(contrast_rng, minval=contrast_range[0], maxval=contrast_range[1]))
        
        # Apply contrast around image mean
        mean = np.mean(image)
        return (image - mean) * factor + mean
    
    def _random_saturation(self, image: np.ndarray, rng: jax.random.PRNGKey) -> np.ndarray:
        """Apply random saturation adjustment."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image  # Skip non-RGB images
            
        saturation_range = self.random_saturation
        rng, sat_rng = jax.random.split(rng)
        factor = float(jax.random.uniform(sat_rng, minval=saturation_range[0], maxval=saturation_range[1]))
        
        # Convert to grayscale for saturation adjustment
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        gray = np.expand_dims(gray, axis=2)
        
        # Blend between grayscale and original based on saturation factor
        return gray + factor * (image - gray)
    
    def _random_hue(self, image: np.ndarray, rng: jax.random.PRNGKey) -> np.ndarray:
        """Apply random hue adjustment."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image  # Skip non-RGB images
            
        hue_range = self.random_hue[0]
        rng, hue_rng = jax.random.split(rng)
        delta = float(jax.random.uniform(hue_rng, minval=-hue_range, maxval=hue_range))
        
        # Simple hue shift by rotating RGB channels
        # This is a simplified version - full HSV conversion would be more accurate
        if abs(delta) > 1e-6:
            # Apply a simple color channel rotation as approximation
            shift_factor = delta * 2.0  # Scale the shift
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            
            # Simple hue shift approximation
            new_r = r + shift_factor * (g - b)
            new_g = g + shift_factor * (b - r)  
            new_b = b + shift_factor * (r - g)
            
            return np.stack([new_r, new_g, new_b], axis=2)
        
        return image


@dataclasses.dataclass(frozen=True)
class ColorJitterTransform(DataTransformFn):
    """
    Simplified color jittering transform for quick color augmentations.
    
    This is a lightweight alternative to ImageAugmentation when you only need
    color-based augmentations without geometric transformations.
    """
    
    brightness: float = 0.2
    contrast: Tuple[float, float] = (0.8, 1.2)
    saturation: Tuple[float, float] = (0.8, 1.2)
    hue: float = 0.05
    prob: float = 0.8
    
    def __call__(self, data: DataDict) -> DataDict:
        """Apply color jittering to images."""
        if "image" not in data:
            return data
            
        # Generate random seed
        seed_val = hash(str(id(data))) % (2**31)
        rng = jax.random.key(seed_val)
        
        # Check if augmentation should be applied
        rng, aug_rng = jax.random.split(rng)
        if jax.random.uniform(aug_rng) > self.prob:
            return data
        
        # Apply to all images
        augmented_images = {}
        for img_key, img in data["image"].items():
            rng, img_rng = jax.random.split(rng)
            augmented_images[img_key] = self._apply_color_jitter(img, img_rng)
        
        augmented_data = dict(data)
        augmented_data["image"] = augmented_images
        return augmented_data
    
    def _apply_color_jitter(self, image: np.ndarray, rng: jax.random.PRNGKey) -> np.ndarray:
        """Apply color jittering to a single image."""
        img = np.asarray(image, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        # Brightness
        rng, bright_rng = jax.random.split(rng)
        brightness_delta = float(jax.random.uniform(bright_rng, minval=-self.brightness, maxval=self.brightness))
        img = img + brightness_delta
        
        # Contrast
        rng, contrast_rng = jax.random.split(rng)
        contrast_factor = float(jax.random.uniform(contrast_rng, minval=self.contrast[0], maxval=self.contrast[1]))
        mean = np.mean(img)
        img = (img - mean) * contrast_factor + mean
        
        # Saturation (if RGB)
        if len(img.shape) == 3 and img.shape[2] == 3:
            rng, sat_rng = jax.random.split(rng)
            sat_factor = float(jax.random.uniform(sat_rng, minval=self.saturation[0], maxval=self.saturation[1]))
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            gray = np.expand_dims(gray, axis=2)
            img = gray + sat_factor * (img - gray)
        
        return np.clip(img, 0.0, 1.0)
