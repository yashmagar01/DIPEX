import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

/// TFLite classifier for plant disease detection
class Classifier {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isLoaded = false;

  // Model configuration
  static const int inputSize = 224;
  static const int numChannels = 3;

  /// Check if the classifier is ready
  bool get isLoaded => _isLoaded;

  /// Get the list of labels
  List<String> get labels => _labels;

  /// Initialize the classifier by loading the model and labels
  Future<bool> loadModel() async {
    try {
      debugPrint('🧠 Classifier: Loading TFLite model...');
      
      // Load the TFLite model from assets
      _interpreter = await Interpreter.fromAsset('model.tflite');
      debugPrint('✅ Classifier: Interpreter created successfully');
      
      // Load labels
      debugPrint('📝 Classifier: Loading labels...');
      final labelsData = await rootBundle.loadString('assets/class_names.txt');
      _labels = labelsData.split('\n')
          .map((label) => label.trim())
          .where((label) => label.isNotEmpty)
          .toList();

      _isLoaded = true;
      debugPrint('✅ Model loaded successfully!');
      debugPrint('📊 Labels: ${_labels.length} classes');
      debugPrint('📊 Classes: ${_labels.join(", ")}');
      return true;
    } catch (e, stackTrace) {
      debugPrint('❌ Error loading model: $e');
      debugPrint('📋 Stack trace: $stackTrace');
      _isLoaded = false;
      return false;
    }
  }

  /// Preprocess image bytes for model input
  Float32List _preprocessImage(img.Image image) {
    // Resize to model input size
    final resized = img.copyResize(image, width: inputSize, height: inputSize);

    // Create input tensor [1, 224, 224, 3]
    final Float32List input = Float32List(1 * inputSize * inputSize * numChannels);
    
    int pixelIndex = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        // Scale to [0, 255] as float (model has rescaling layer that handles normalization)
        input[pixelIndex++] = pixel.r.toDouble();
        input[pixelIndex++] = pixel.g.toDouble();
        input[pixelIndex++] = pixel.b.toDouble();
      }
    }

    return input;
  }

  /// Run inference on image bytes (from captured photo)
  Future<Map<String, dynamic>> classifyImageBytes(Uint8List imageBytes) async {
    if (!_isLoaded || _interpreter == null) {
      debugPrint('❌ Model not loaded!');
      return {'label': 'Model not loaded', 'confidence': 0.0};
    }

    try {
      debugPrint('🔍 Starting image classification...');
      
      // Decode image
      final image = img.decodeImage(imageBytes);
      if (image == null) {
        debugPrint('❌ Failed to decode image');
        return {'label': 'Invalid image', 'confidence': 0.0};
      }
      
      debugPrint('📐 Image size: ${image.width}x${image.height}');

      // Preprocess
      final input = _preprocessImage(image);
      final inputTensor = input.reshape([1, inputSize, inputSize, numChannels]);

      // Prepare output tensor
      final output = List.filled(1 * _labels.length, 0.0).reshape([1, _labels.length]);

      // Run inference
      debugPrint('🧠 Running inference...');
      _interpreter!.run(inputTensor, output);

      // Get prediction
      final List<double> probabilities = List<double>.from(output[0]);
      
      int maxIndex = 0;
      double maxProb = probabilities[0];
      for (int i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIndex = i;
        }
      }

      debugPrint('✅ Prediction: ${_labels[maxIndex]} (${(maxProb * 100).toStringAsFixed(1)}%)');

      return {
        'label': _labels[maxIndex],
        'confidence': maxProb,
        'index': maxIndex,
      };
    } catch (e, stackTrace) {
      debugPrint('❌ Inference error: $e');
      debugPrint('📋 Stack: $stackTrace');
      return {'label': 'Error: $e', 'confidence': 0.0};
    }
  }

  /// Run inference on a file path
  Future<Map<String, dynamic>> classifyFile(String filePath) async {
    try {
      final bytes = await File(filePath).readAsBytes();
      return classifyImageBytes(bytes);
    } catch (e) {
      debugPrint('❌ File read error: $e');
      return {'label': 'File error', 'confidence': 0.0};
    }
  }

  /// Dispose of resources
  void dispose() {
    _interpreter?.close();
    _isLoaded = false;
  }
}
