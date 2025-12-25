import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'classifier.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Get available cameras
  try {
    cameras = await availableCameras();
    debugPrint('📷 Found ${cameras.length} cameras');
  } catch (e) {
    debugPrint('❌ Error getting cameras: $e');
  }
  
  runApp(const EdgeAIAgronomistApp());
}

class EdgeAIAgronomistApp extends StatelessWidget {
  const EdgeAIAgronomistApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Edge-AI Agronomist',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.green,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  CameraController? _cameraController;
  final Classifier _classifier = Classifier();
  
  bool _isCameraInitialized = false;
  bool _isModelLoaded = false;
  bool _isProcessing = false;
  String _prediction = '';
  double _confidence = 0.0;
  String _statusMessage = 'Initializing...';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }
    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
      _isCameraInitialized = false;
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  Future<void> _initialize() async {
    debugPrint('🚀 Starting initialization...');
    
    // Step 1: Request camera permission
    setState(() => _statusMessage = 'Requesting camera permission...');
    final status = await Permission.camera.request();
    debugPrint('📱 Permission status: $status');
    
    if (!status.isGranted) {
      setState(() {
        _statusMessage = 'Camera permission denied';
      });
      return;
    }

    // Step 2: Load the TFLite model
    setState(() => _statusMessage = 'Loading AI model...');
    final modelLoaded = await _classifier.loadModel();
    
    if (modelLoaded) {
      setState(() {
        _isModelLoaded = true;
        _statusMessage = 'Model loaded! Initializing camera...';
      });
    } else {
      setState(() {
        _statusMessage = 'Failed to load AI model';
      });
      // Continue anyway - we can still show camera preview
    }

    // Step 3: Initialize camera
    await _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    debugPrint('📷 Initializing camera...');
    
    if (cameras.isEmpty) {
      setState(() => _statusMessage = 'No cameras found');
      return;
    }

    // Dispose old controller if exists
    if (_cameraController != null) {
      await _cameraController!.dispose();
    }

    _cameraController = CameraController(
      cameras[0],
      ResolutionPreset.high,
      enableAudio: false,
    );

    try {
      await _cameraController!.initialize();
      debugPrint('✅ Camera initialized');
      
      if (!mounted) return;
      
      setState(() {
        _isCameraInitialized = true;
        _statusMessage = _isModelLoaded 
            ? 'Ready! Tap the button to analyze'
            : 'Camera ready (Model not loaded)';
      });
      
    } catch (e) {
      debugPrint('❌ Camera error: $e');
      setState(() {
        _statusMessage = 'Camera error: $e';
        _isCameraInitialized = false;
      });
    }
  }

  /// Capture photo and run inference
  Future<void> _captureAndAnalyze() async {
    if (!_isCameraInitialized || !_isModelLoaded || _isProcessing) {
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusMessage = 'Capturing image...';
      _prediction = '';
      _confidence = 0.0;
    });

    try {
      // Capture image
      debugPrint('📸 Capturing image...');
      final XFile image = await _cameraController!.takePicture();
      debugPrint('📸 Image captured: ${image.path}');
      
      setState(() => _statusMessage = 'Analyzing...');

      // Read image bytes and classify
      final bytes = await image.readAsBytes();
      final result = await _classifier.classifyImageBytes(bytes);

      if (mounted) {
        setState(() {
          _prediction = result['label'] as String;
          _confidence = (result['confidence'] as double) * 100;
          _statusMessage = 'Analysis complete!';
          _isProcessing = false;
        });
      }
    } catch (e) {
      debugPrint('❌ Capture error: $e');
      if (mounted) {
        setState(() {
          _statusMessage = 'Error: $e';
          _isProcessing = false;
        });
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Camera Preview
          if (_isCameraInitialized && _cameraController != null && _cameraController!.value.isInitialized)
            Positioned.fill(
              child: CameraPreview(_cameraController!),
            )
          else
            Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(color: Colors.green),
                  const SizedBox(height: 20),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 40),
                    child: Text(
                      _statusMessage,
                      style: const TextStyle(color: Colors.white70, fontSize: 14),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ],
              ),
            ),

          // Top App Bar
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.only(
                top: MediaQuery.of(context).padding.top + 10,
                left: 20,
                right: 20,
                bottom: 10,
              ),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.black.withOpacity(0.7),
                    Colors.transparent,
                  ],
                ),
              ),
              child: Row(
                children: [
                  const Icon(Icons.eco, color: Colors.green, size: 28),
                  const SizedBox(width: 10),
                  const Text(
                    'Edge-AI Agronomist',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const Spacer(),
                  // Status indicator
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: _isModelLoaded 
                          ? Colors.green.withOpacity(0.3) 
                          : Colors.orange.withOpacity(0.3),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            color: _isModelLoaded ? Colors.green : Colors.orange,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 6),
                        Text(
                          _isModelLoaded ? 'READY' : 'LOADING',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Bottom Panel with Capture Button and Results
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.only(
                bottom: MediaQuery.of(context).padding.bottom + 20,
                top: 20,
                left: 20,
                right: 20,
              ),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.bottomCenter,
                  end: Alignment.topCenter,
                  colors: [
                    Colors.black.withOpacity(0.95),
                    Colors.black.withOpacity(0.8),
                    Colors.transparent,
                  ],
                ),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Result display (only show when we have a prediction)
                  if (_prediction.isNotEmpty) ...[
                    // Confidence bar
                    Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: Row(
                        children: [
                          Expanded(
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(4),
                              child: LinearProgressIndicator(
                                value: _confidence / 100,
                                backgroundColor: Colors.grey[800],
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  _confidence >= 60 ? Colors.green : Colors.orange,
                                ),
                                minHeight: 6,
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Text(
                            '${_confidence.toStringAsFixed(1)}%',
                            style: TextStyle(
                              color: _confidence >= 60 ? Colors.green : Colors.orange,
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                    
                    // Prediction result box
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 20),
                      margin: const EdgeInsets.only(bottom: 20),
                      decoration: BoxDecoration(
                        color: _confidence >= 60
                            ? Colors.green.withOpacity(0.2)
                            : Colors.grey.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(
                          color: _confidence >= 60
                              ? Colors.green.withOpacity(0.5)
                              : Colors.grey.withOpacity(0.3),
                          width: 1,
                        ),
                      ),
                      child: Column(
                        children: [
                          Icon(
                            _confidence >= 60 ? Icons.check_circle : Icons.warning_amber,
                            color: _confidence >= 60 ? Colors.green : Colors.orange,
                            size: 32,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            _formatLabel(_prediction),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _prediction.toLowerCase().contains('healthy')
                                ? '✓ Plant appears healthy'
                                : '⚠ Disease detected - consult an expert',
                            style: TextStyle(
                              color: _prediction.toLowerCase().contains('healthy')
                                  ? Colors.green[300]
                                  : Colors.orange[300],
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                  
                  // Capture button
                  GestureDetector(
                    onTap: (_isCameraInitialized && _isModelLoaded && !_isProcessing)
                        ? _captureAndAnalyze
                        : null,
                    child: Container(
                      width: 80,
                      height: 80,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: _isProcessing
                            ? Colors.grey
                            : (_isModelLoaded ? Colors.green : Colors.grey),
                        border: Border.all(
                          color: Colors.white.withOpacity(0.5),
                          width: 4,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.green.withOpacity(0.3),
                            blurRadius: 20,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                      child: Center(
                        child: _isProcessing
                            ? const SizedBox(
                                width: 30,
                                height: 30,
                                child: CircularProgressIndicator(
                                  color: Colors.white,
                                  strokeWidth: 3,
                                ),
                              )
                            : const Icon(
                                Icons.camera_alt,
                                color: Colors.white,
                                size: 36,
                              ),
                      ),
                    ),
                  ),
                  
                  const SizedBox(height: 12),
                  
                  // Status text
                  Text(
                    _isProcessing 
                        ? _statusMessage 
                        : (_isModelLoaded 
                            ? 'Tap to analyze plant leaf' 
                            : _statusMessage),
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.7),
                      fontSize: 14,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// Format the label for display (replace underscores, capitalize)
  String _formatLabel(String label) {
    return label
        .replaceAll('___', ' - ')
        .replaceAll('__', ' - ')
        .replaceAll('_', ' ')
        .split(' ')
        .map((word) => word.isNotEmpty
            ? '${word[0].toUpperCase()}${word.substring(1).toLowerCase()}'
            : '')
        .join(' ');
  }
}
