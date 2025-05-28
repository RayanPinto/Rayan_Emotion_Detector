# Rayan_Emotion_Detector
Real-time facial feature &amp; emotion detection by Rayan
Facial Feature and Emotion Detection

A real-time facial feature and emotion detection system developed by Rayan. This project leverages computer vision and deep learning to analyze webcam video feeds, detecting faces, eyes, and smiles using OpenCV's Haar cascade classifiers and recognizing emotions (e.g., Happy, Sad, Neutral, Angry, Fear, Surprise, Disgust) with DeepFace. The system logs detected emotions and generates visualizations (bar and pie charts) to provide insights into emotional patterns, making it suitable for applications in human-computer interaction, sentiment analysis, and behavioral studies.

Project Overview

The project processes live webcam footage to identify facial features and classify emotions in real-time. It combines traditional computer vision techniques (OpenCV) with deep learning (DeepFace) to achieve robust detection and analysis. Key outputs include:





Real-time display of detected faces, eyes, smiles, and emotions with confidence scores and FPS (frames per second).



A log file (emotions.txt) recording the count of each detected emotion.



Bar and pie charts (emotion_bar_chart.png, emotion_pie_chart.png) visualizing emotion distributions, saved to the output folder.

The system is optimized with dynamic parameters (e.g., scale factor, minimum neighbors) based on frame brightness and includes frame skipping to balance performance and accuracy. Each run clears previous logs and charts, ensuring fresh data is stored in the output folder.

Features





Facial Feature Detection: Identifies faces, eyes, and smiles using OpenCV's Haar cascade classifiers with confidence scoring.



Emotion Recognition: Classifies emotions using DeepFace's deep neural network, supporting seven emotions: Happy, Sad, Neutral, Angry, Fear, Surprise, and Disgust.



Real-Time Processing: Displays live webcam feed with annotated detections and FPS.



Emotion Logging: Records emotion counts in output/emotions.txt for analysis.



Data Visualization: Generates bar and pie charts with Matplotlib, saved as output/emotion_bar_chart.png and output/emotion_pie_chart.png.



Output Management: Automatically clears previous logs and charts before saving new ones.



Dynamic Adjustments: Adapts detection parameters based on lighting conditions for improved accuracy.

Technologies





Python 3.11+: Core programming language.



OpenCV 4.10.0: For facial feature detection using Haar cascades.



DeepFace 0.0.93: For emotion recognition with deep learning.



TensorFlow 2.18.0: Backend for DeepFace's neural network.



Matplotlib 3.8.0: For generating bar and pie charts.



NumPy 1.26.0: For numerical computations
