
3D PRINTING DEFECT DETECTION USING YOLO26
<img width="568" height="2368" alt="image" src="https://github.com/user-attachments/assets/2b1d3e33-7679-4a0a-85c4-80ab43f482b8" />

PROJECT OVERVIEW
This project aims to develop an automated defect detection system for Fused Deposition Modeling (FDM) 3D printing using computer vision and deep learning. The system is designed to identify common printing defects from images of printed parts and classify them into predefined defect categories.

The project utilizes the YOLO26 object detection model to detect and classify defects directly from images. The final goal is to integrate the model into a defect inspection workflow and potentially extend it into an intelligent troubleshooting chatbot using Ollama and computer vision.

Qwen3:8B is used for responsive human-like response for ease of comprehension and allows ineteraction to ease the troubleshooting procress.

OBJECTIVES
- Detect defects in 3D printed parts automatically.
- Classify detected defects into common failure categories.
- Reduce manual inspection effort.
- Provide a foundation for future automated troubleshooting and quality control systems.

DEFECT CLASSES
- Crack: Visible cracks or fractures in printed parts.
- Spaghetti: Severe printing failure where filament extrudes uncontrollably.
- Stringing: Thin filament strings between printed features.

DATASET
The dataset consists of labeled images organized in YOLO format with train/images, train/labels, valid/images, valid/labels, and data.yaml.

MODEL CONFIGURATION
Base Model: yolo26n.pt

Training Parameters:
- Epochs: 500
- Image Size: 640x640
- Batch Size: Auto
- Optimizer: Auto
- Device: GPU (CUDA)
- Learning Rate: 0.01
- Momentum: 0.937

EVALUATION METRICS
- Precision
- Recall
- F1 Score
- Precision-Recall Curve
- Mean Average Precision (mAP@0.5)
- Confusion Matrix

RESULTS

Overall Performance:
- mAP@0.5: 0.745
- Best F1 Score: 0.75
- Optimal Confidence Threshold: 0.628

Per-Class Performance:
- Crack: AP@0.5 = 0.940
- Spaghetti: AP@0.5 = 0.500
- Stringing: AP@0.5 = 0.793

Observations:
Crack Detection:
- Highest performing class.
- AP@0.5 = 0.940.
- Precision consistently above 90%.
- Very few missed detections.

Stringing Detection:
- Good performance.
- AP@0.5 = 0.793.
- High precision and recall.
- Some confusion with background samples.

Spaghetti Detection:
- Weakest performing class.
- AP@0.5 = 0.500.
- Lower precision and recall.
- Likely caused by limited data and greater visual variation.

CONFUSION MATRIX ANALYSIS
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/0743cf23-3f0a-4479-9360-0506e118fc7c" />
Correct Detections:
- Crack: 141
- Spaghetti: 15
- Stringing: 138

Missed as Background:
- Crack: 9
- Spaghetti: 13
- Stringing: 39

Normalized Results:
- Crack: 94% correctly detected, 6% missed.
- Spaghetti: 54% correctly detected, 46% missed.
- Stringing: 78% correctly detected, 22% missed.

F1 SCORE ANALYSIS
<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/ee383279-79a3-4585-a9d5-823f9a1e6fd5" />

Maximum F1 Score = 0.75
Confidence Threshold = 0.628

This threshold provides the best balance between precision and recall and is recommended for deployment.

STRENGTHS
- Excellent Crack detection performance.
- Strong Stringing classification accuracy.
- High overall precision.
- Suitable for real-time inspection applications.
- Lightweight YOLO26 model enables fast inference.

LIMITATIONS
- Lower performance on Spaghetti defects.
- Some defects are missed and classified as background.
- Dataset imbalance may affect class performance.
- Detection accuracy depends on image quality and lighting conditions.

FUTURE WORK
- Collect additional Spaghetti defect samples.
- Increase dataset diversity.
- Improve data augmentation.
- Experiment with larger YOLO26 variants.
- Integrate a real-time inspection dashboard.
- Develop an Ollama-powered troubleshooting chatbot.

CONCLUSION
A YOLO26-based object detection model was successfully developed for identifying common 3D printing defects. The model achieved an overall mAP@0.5 of 74.5% and demonstrated excellent performance in detecting Crack defects (AP = 0.940) and strong performance for Stringing defects (AP = 0.793). However, Spaghetti defects (AP = 0.500) remain the most challenging class due to lower recall and higher rates of missed detections.

The confusion matrix shows that most errors originate from defects being classified as background rather than being confused with other defect classes. This indicates that future improvements should focus on increasing detection sensitivity, particularly for Spaghetti defects.

Overall, the model demonstrates that deep-learning-based vision systems can effectively support automated quality inspection in additive manufacturing environments and provides a strong foundation for future intelligent defect diagnosis and troubleshooting systems.

