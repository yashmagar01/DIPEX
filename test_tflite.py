"""
Edge-AI Agronomist - TFLite Model Testing Script
=================================================
This script tests the trained TFLite model on a single image.

Usage:
    1. Place your test image in the DIPEX folder
    2. Update IMAGE_PATH below with your image filename
    3. Run: py test_tflite.py
"""

import numpy as np
from PIL import Image
import tensorflow as tf

# ============================================================================
# CONFIGURATION - Change this to your test image
# ============================================================================

IMAGE_PATH = "test_tomato.jpg"  # Change this to your test image filename

# Model and labels paths
MODEL_PATH = "model.tflite"
LABELS_PATH = "class_names.txt"

# Input configuration
IMG_SIZE = (224, 224)

# ============================================================================
# LOAD CLASS NAMES
# ============================================================================

def load_class_names(labels_path):
    """Load class names from the text file."""
    with open(labels_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image_path, target_size):
    """
    Load and preprocess the image for inference.
    
    Note: The TFLite model contains the Rescaling layer, so we pass
    raw pixel values [0-255] as float32.
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if necessary (handles PNG with alpha, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to target size
    img = img.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ============================================================================
# TFLITE INFERENCE
# ============================================================================

def run_inference(model_path, image_array):
    """
    Run inference using the TFLite interpreter.
    """
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor (probabilities)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data[0]  # Return first (and only) batch result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "🌿" * 25)
    print("  EDGE-AI AGRONOMIST - MODEL TESTER")
    print("🌿" * 25 + "\n")
    
    # Load class names
    print(f"📂 Loading labels from: {LABELS_PATH}")
    class_names = load_class_names(LABELS_PATH)
    print(f"✅ Loaded {len(class_names)} classes\n")
    
    # Preprocess image
    print(f"🖼️  Loading image: {IMAGE_PATH}")
    try:
        img_array = preprocess_image(IMAGE_PATH, IMG_SIZE)
        print(f"✅ Image preprocessed: {img_array.shape}")
        print(f"   Pixel range: [{img_array.min():.1f}, {img_array.max():.1f}]")
    except FileNotFoundError:
        print(f"❌ ERROR: Image not found: {IMAGE_PATH}")
        print(f"   Please place your test image in the current directory.")
        return
    
    # Run inference
    print(f"\n🔄 Running inference with: {MODEL_PATH}")
    try:
        predictions = run_inference(MODEL_PATH, img_array)
    except Exception as e:
        print(f"❌ ERROR during inference: {str(e)}")
        return
    
    # Get the predicted class
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index] * 100
    predicted_class = class_names[predicted_index]
    
    # Display results
    print("\n" + "="*50)
    print("  🎯 PREDICTION RESULT")
    print("="*50)
    print(f"\n   Predicted Class: {predicted_class}")
    print(f"   Confidence: {confidence:.2f}%")
    
    # Confidence check
    if confidence < 50:
        print(f"\n   ⚠️  Low Confidence - Image might be unclear or not a plant disease.")
    else:
        print(f"\n   ✅ High confidence prediction!")
    
    # Show top 3 predictions
    print("\n" + "-"*50)
    print("  📊 Top 3 Predictions:")
    print("-"*50)
    
    top_indices = np.argsort(predictions)[-3:][::-1]
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i}. {class_names[idx]}: {predictions[idx]*100:.2f}%")
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
