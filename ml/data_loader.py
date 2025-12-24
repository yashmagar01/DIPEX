"""
Plant Disease Detection - Data Loading Script
==============================================
This script loads the PlantVillage dataset for training a MobileNetV2-based model.

Usage:
    1. If you have pre-split train/val folders:
       - Set DATASET_PATH to your dataset directory containing 'train' and 'val' subfolders
       
    2. If your data is in a single folder with class subfolders (like PlantVillage):
       - Set DATASET_PATH to that folder
       - The script will automatically split it into train/val (80/20)

Author: Plant Disease Detection Project
Date: December 2024
"""

import tensorflow as tf
from tensorflow import keras
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your dataset
# Option 1: Single folder with class subfolders (will auto-split)
DATASET_PATH = r"c:\Users\Yash\OneDrive\Desktop\DIPEX\PlantVillage"

# Option 2: If you already have train/val split, set these paths
# TRAIN_PATH = r"c:\Users\Yash\OneDrive\Desktop\DIPEX\dataset\train"
# VAL_PATH = r"c:\Users\Yash\OneDrive\Desktop\DIPEX\dataset\val"

# Image configuration
IMG_SIZE = (224, 224)  # Standard size for MobileNetV2
BATCH_SIZE = 32        # Adjust based on your GPU memory
SEED = 42              # For reproducibility

# Validation split ratio (only used if auto-splitting)
VALIDATION_SPLIT = 0.2

# ============================================================================
# NORMALIZATION LAYER FOR MOBILENETV2
# ============================================================================

def get_normalization_layer(mode='tf'):
    """
    Returns a normalization layer for MobileNetV2.
    
    MobileNetV2 expects pixel values in the range [-1, 1] when using
    tf.keras.applications.mobilenet_v2.preprocess_input.
    
    Args:
        mode: 'tf' for [-1, 1] range (recommended for MobileNetV2)
              'standard' for [0, 1] range
    
    Returns:
        A normalization layer or preprocessing function
    """
    if mode == 'tf':
        # MobileNetV2 preprocessing: scales pixels to [-1, 1]
        # This is the recommended approach for transfer learning
        return tf.keras.layers.Rescaling(1./127.5, offset=-1)
    else:
        # Standard normalization to [0, 1]
        return tf.keras.layers.Rescaling(1./255)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_dataset_with_split(dataset_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE, 
                            validation_split=VALIDATION_SPLIT, seed=SEED):
    """
    Loads dataset from a single folder and automatically splits into train/val.
    
    Args:
        dataset_path: Path to the dataset folder containing class subfolders
        img_size: Target image size (height, width)
        batch_size: Number of images per batch
        validation_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        train_ds, val_ds: Training and validation datasets
        class_names: List of class names
    """
    print(f"\n{'='*60}")
    print("📁 Loading dataset from: {dataset_path}")
    print(f"{'='*60}\n")
    
    # Load training dataset (80% of data)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',  # One-hot encoded labels for multi-class
        shuffle=True
    )
    
    # Load validation dataset (20% of data)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False  # No shuffling for validation
    )
    
    # Get class names
    class_names = train_ds.class_names
    
    print(f"✅ Found {len(class_names)} classes:")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    
    return train_ds, val_ds, class_names


def load_dataset_from_split_folders(train_path, val_path, img_size=IMG_SIZE, 
                                     batch_size=BATCH_SIZE):
    """
    Loads dataset from pre-split train and val folders.
    
    Args:
        train_path: Path to training data folder
        val_path: Path to validation data folder
        img_size: Target image size (height, width)
        batch_size: Number of images per batch
    
    Returns:
        train_ds, val_ds: Training and validation datasets
        class_names: List of class names
    """
    print(f"\n{'='*60}")
    print("📁 Loading from separate train/val folders")
    print(f"{'='*60}\n")
    
    # Load training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
        seed=SEED
    )
    
    # Load validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    
    class_names = train_ds.class_names
    
    print(f"✅ Found {len(class_names)} classes:")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    
    return train_ds, val_ds, class_names


# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

def configure_for_performance(ds, normalize=True, augment=False, cache=True):
    """
    Configures the dataset for optimal performance.
    
    Uses the following optimizations:
    - Caching: Stores dataset in memory after first epoch
    - Prefetching: Prepares next batch while current batch is being processed
    - Normalization: Scales pixel values to [-1, 1] for MobileNetV2
    
    Args:
        ds: TensorFlow dataset
        normalize: Whether to apply MobileNetV2 normalization
        augment: Whether to apply data augmentation (for training only)
        cache: Whether to cache the dataset
    
    Returns:
        Optimized dataset
    """
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Cache the dataset
    if cache:
        ds = ds.cache()
    
    # Normalize for MobileNetV2 (pixels to [-1, 1])
    if normalize:
        normalization_layer = get_normalization_layer(mode='tf')
        ds = ds.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=AUTOTUNE
        )
    
    # Prefetch next batch while training on current batch
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds


# ============================================================================
# DATA AUGMENTATION (Optional)
# ============================================================================

def get_augmentation_layer():
    """
    Returns a data augmentation layer for training.
    
    Augmentations help prevent overfitting by creating variations of images.
    Apply this only to the training dataset, not validation.
    
    Returns:
        Sequential model with augmentation layers
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.2),
        keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")


def apply_augmentation(train_ds):
    """
    Applies data augmentation to the training dataset.
    
    Args:
        train_ds: Training dataset
    
    Returns:
        Augmented training dataset
    """
    AUTOTUNE = tf.data.AUTOTUNE
    augmentation = get_augmentation_layer()
    
    return train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to load and prepare the dataset.
    """
    print("\n" + "🌿" * 30)
    print("  PLANT DISEASE DETECTION - DATA LOADER")
    print("🌿" * 30 + "\n")
    
    # Check TensorFlow version and GPU availability
    print(f"📦 TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 GPU(s) Available: {len(gpus)}")
        for gpu in gpus:
            print(f"   - {gpu.name}")
    else:
        print("⚠️  No GPU detected. Training will use CPU.")
    
    print()
    
    # -------------------------------------------------------------------------
    # OPTION 1: Load from single folder with auto-split (CURRENT SETUP)
    # -------------------------------------------------------------------------
    train_ds, val_ds, class_names = load_dataset_with_split(
        dataset_path=DATASET_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )
    
    # -------------------------------------------------------------------------
    # OPTION 2: Load from separate train/val folders (uncomment if needed)
    # -------------------------------------------------------------------------
    # train_ds, val_ds, class_names = load_dataset_from_split_folders(
    #     train_path=TRAIN_PATH,
    #     val_path=VAL_PATH,
    #     img_size=IMG_SIZE,
    #     batch_size=BATCH_SIZE
    # )
    
    # -------------------------------------------------------------------------
    # CONFIGURE FOR PERFORMANCE
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("⚡ Configuring datasets for optimal performance...")
    print(f"{'='*60}\n")
    
    # Apply data augmentation to training data (optional but recommended)
    train_ds_augmented = apply_augmentation(train_ds)
    
    # Configure for performance (caching, normalization, prefetching)
    train_ds_final = configure_for_performance(
        train_ds_augmented, 
        normalize=True, 
        cache=True
    )
    
    val_ds_final = configure_for_performance(
        val_ds, 
        normalize=True, 
        cache=True
    )
    
    # -------------------------------------------------------------------------
    # DATASET INFO
    # -------------------------------------------------------------------------
    print("📊 Dataset Information:")
    print(f"   • Image Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"   • Batch Size: {BATCH_SIZE}")
    print(f"   • Number of Classes: {len(class_names)}")
    print(f"   • Training Batches: {tf.data.experimental.cardinality(train_ds_final).numpy()}")
    print(f"   • Validation Batches: {tf.data.experimental.cardinality(val_ds_final).numpy()}")
    print(f"   • Normalization: [-1, 1] (MobileNetV2 compatible)")
    print(f"   • Prefetching: Enabled (AUTOTUNE)")
    print(f"   • Caching: Enabled")
    
    # -------------------------------------------------------------------------
    # VERIFY A SAMPLE BATCH
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("🔍 Verifying sample batch...")
    print(f"{'='*60}\n")
    
    for images, labels in train_ds_final.take(1):
        print(f"   • Batch shape: {images.shape}")
        print(f"   • Labels shape: {labels.shape}")
        print(f"   • Pixel value range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
        print(f"   • Data type: {images.dtype}")
    
    print("\n" + "✅" * 30)
    print("  DATASET LOADED SUCCESSFULLY!")
    print("✅" * 30 + "\n")
    
    return train_ds_final, val_ds_final, class_names


# ============================================================================
# HELPER FUNCTION FOR MODEL INTEGRATION
# ============================================================================

def get_datasets():
    """
    Convenience function to get train and validation datasets.
    Call this from your training script.
    
    Returns:
        train_ds: Preprocessed training dataset
        val_ds: Preprocessed validation dataset
        class_names: List of class names
        num_classes: Number of classes
    """
    train_ds, val_ds, class_names = main()
    return train_ds, val_ds, class_names, len(class_names)


# Run the script
if __name__ == "__main__":
    train_ds, val_ds, class_names = main()
    
    # Save class names for later use
    print("\n📝 Saving class names to 'class_names.txt'...")
    with open(r"c:\Users\Yash\OneDrive\Desktop\DIPEX\class_names.txt", 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print("✅ Class names saved!")
