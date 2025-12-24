"""
Edge-AI Agronomist - Plant Disease Detection Model Training Script
====================================================================
This script trains a MobileNetV2-based model for plant disease classification
and converts it to TensorFlow Lite for mobile deployment.

Author: Edge-AI Agronomist Project
Date: December 2024
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset path (using the nested path as specified)
DATASET_PATH = r"C:\Users\Yash\OneDrive\Desktop\DIPEX\PlantVillage\PlantVillage"

# Image and training configuration
IMG_SIZE = (224, 224)  # MobileNetV2 standard input size
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42

# Output paths
MODEL_SAVE_PATH = r"C:\Users\Yash\OneDrive\Desktop\DIPEX\plant_disease_model.h5"
TFLITE_SAVE_PATH = r"C:\Users\Yash\OneDrive\Desktop\DIPEX\model.tflite"
CLASS_NAMES_PATH = r"C:\Users\Yash\OneDrive\Desktop\DIPEX\class_names.txt"

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

def load_data():
    """
    Load and prepare the dataset using image_dataset_from_directory.
    Splits data 80/20 for training/validation.
    """
    print("\n" + "="*60)
    print("  STEP 1: DATA LOADING")
    print("="*60 + "\n")
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")
    
    print(f"📁 Loading dataset from: {DATASET_PATH}")
    print(f"📐 Resizing images to: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"✂️  Validation split: {VALIDATION_SPLIT * 100}%\n")
    
    # Load training dataset (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'  # For SparseCategoricalCrossentropy
    )
    
    # Load validation dataset (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    # Get class names
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    # Print class names (as requested)
    print("🌱 CLASS NAMES DETECTED:")
    print("-" * 40)
    for i, name in enumerate(class_names):
        print(f"   {i:2d}: {name}")
    print("-" * 40)
    print(f"\n✅ Total Classes: {num_classes}")
    
    # Save class names to file for later use
    with open(CLASS_NAMES_PATH, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"📝 Class names saved to: {CLASS_NAMES_PATH}")
    
    return train_ds, val_ds, class_names, num_classes


# ============================================================================
# STEP 2: MODEL ARCHITECTURE (TRANSFER LEARNING)
# ============================================================================

def build_model(num_classes):
    """
    Build the MobileNetV2-based transfer learning model.
    """
    print("\n" + "="*60)
    print("  STEP 2: MODEL ARCHITECTURE")
    print("="*60 + "\n")
    
    # Create preprocessing layer for MobileNetV2 (scales to [-1, 1])
    preprocessing_layer = layers.Rescaling(1./127.5, offset=-1)
    
    # Load MobileNetV2 base model (pre-trained on ImageNet)
    print("📥 Loading MobileNetV2 with ImageNet weights...")
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model (transfer learning)
    base_model.trainable = False
    print("🔒 Base model frozen (trainable=False)")
    
    # Build the complete model
    print("🔧 Adding custom classification layers...")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        
        # Preprocessing (normalize to [-1, 1] for MobileNetV2)
        preprocessing_layer,
        
        # Base model (frozen)
        base_model,
        
        # Custom classification head
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print("\n📊 MODEL SUMMARY:")
    print("-" * 40)
    model.summary()
    
    return model


# ============================================================================
# STEP 3: COMPILATION & TRAINING
# ============================================================================

def compile_and_train(model, train_ds, val_ds):
    """
    Compile the model and train for the specified epochs.
    """
    print("\n" + "="*60)
    print("  STEP 3: COMPILATION & TRAINING")
    print("="*60 + "\n")
    
    # Compile the model
    print(f"⚙️  Optimizer: Adam (learning_rate={LEARNING_RATE})")
    print(f"📉 Loss: SparseCategoricalCrossentropy")
    print(f"📈 Metrics: accuracy\n")
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Train the model
    print(f"🚀 Starting training for {EPOCHS} epochs...")
    print("-" * 40)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    
    print("\n✅ Training complete!")
    
    # Print final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\n📊 Final Training Accuracy: {final_train_acc:.4f}")
    print(f"📊 Final Validation Accuracy: {final_val_acc:.4f}")
    
    # Save the model
    print(f"\n💾 Saving model to: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model saved successfully!")
    
    return model, history


# ============================================================================
# STEP 4: TFLITE CONVERSION
# ============================================================================

def convert_to_tflite(model):
    """
    Convert the trained Keras model to TensorFlow Lite format.
    """
    print("\n" + "="*60)
    print("  STEP 4: TFLITE CONVERSION")
    print("="*60 + "\n")
    
    print("🔄 Converting model to TensorFlow Lite format...")
    
    try:
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optional: Enable optimizations for smaller model size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(TFLITE_SAVE_PATH, 'wb') as f:
            f.write(tflite_model)
        
        # Get file size
        tflite_size = os.path.getsize(TFLITE_SAVE_PATH) / (1024 * 1024)
        
        print(f"✅ TFLite model saved to: {TFLITE_SAVE_PATH}")
        print(f"📦 TFLite model size: {tflite_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error during TFLite conversion: {str(e)}")
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to orchestrate the entire training pipeline.
    """
    print("\n" + "🌿" * 30)
    print("  EDGE-AI AGRONOMIST - MODEL TRAINING")
    print("🌿" * 30 + "\n")
    
    # Check TensorFlow version and GPU
    print(f"📦 TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 GPU(s) Available: {len(gpus)}")
        for gpu in gpus:
            print(f"   - {gpu.name}")
    else:
        print("⚠️  No GPU detected. Training will use CPU (this may be slow).")
    
    # Step 1: Load Data
    train_ds, val_ds, class_names, num_classes = load_data()
    
    # Step 2: Build Model
    model = build_model(num_classes)
    
    # Step 3: Compile and Train
    model, history = compile_and_train(model, train_ds, val_ds)
    
    # Step 4: Convert to TFLite
    convert_to_tflite(model)
    
    # Final Summary
    print("\n" + "="*60)
    print("  🎉 TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n📁 Output Files:")
    print(f"   • Keras Model:  {MODEL_SAVE_PATH}")
    print(f"   • TFLite Model: {TFLITE_SAVE_PATH}")
    print(f"   • Class Names:  {CLASS_NAMES_PATH}")
    print(f"\n🌱 Classes Trained: {num_classes}")
    print(f"📊 Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")
    print("\n" + "✅" * 30 + "\n")


if __name__ == "__main__":
    main()
