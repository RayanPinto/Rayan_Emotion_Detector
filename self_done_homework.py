import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from deepface import DeepFace

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress deprecation warnings

# Initialize emotion counts
emotion_counts = {'Happy': 0, 'Sad': 0, 'Neutral': 0, 'Angry': 0, 'Fear': 0, 'Surprise': 0, 'Disgust': 0}

# Create output directory
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load Cascade Classifiers
def load_cascade(cascade_path):
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise IOError(f"Failed to load cascade classifier: {cascade_path}")
    return cascade

face_cascade = load_cascade('haarcascade_frontalface_default.xml')
eye_cascade = load_cascade('haarcascade_eye.xml')
smile_cascade = load_cascade('haarcascade_smile.xml')

# Calculate Eye Aspect Ratio (EAR) for eye openness (simplified heuristic)
def calculate_ear(eye_roi):
    h, w = eye_roi.shape[:2]
    return w / h if h > 0 else 0

# Detection function with emotion estimation
def detect(gray, frame, scale_factor=1.15, min_neighbors=5):
    global emotion_counts
    start_time = time.time()
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, 
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        confidence = min(1.0, len(faces) * 0.1 + w * h / (gray.shape[0] * gray.shape[1]))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'Face: {confidence:.2f}', (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Emotion detection with DeepFace
        try:
            face_img = frame[y:y+h, x:x+w]
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion'].capitalize()
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        except:
            emotion = "Neutral"
            emotion_counts['Neutral'] += 1
        
        cv2.putText(frame, f'Emotion: {emotion}', (x, y+h+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=22, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            eye_confidence = min(1.0, ew * eh / (w * h))
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(roi_color, f'Eye: {eye_confidence:.2f}', 
                       (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=25, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            smile_confidence = min(1.0, sw * sh / (w * h))
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            cv2.putText(roi_color, f'Smile: {smile_confidence:.2f}', 
                       (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame

# Generate and save emotion charts
def save_emotion_charts():
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    
    # Bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(emotions, counts, color=['#36A2EB', '#FF6384', '#FFCE56', '#FF5733', '#C70039', '#900C3F', '#581845'])
    plt.title('Detected Emotions Over Time')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_bar_chart.png'))
    plt.close()
    
    # Pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=emotions, colors=['#36A2EB', '#FF6384', '#FFCE56', '#FF5733', '#C70039', '#900C3F', '#581845'],
            autopct='%1.1f%%', startangle=140)
    plt.title('Emotion Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_pie_chart.png'))
    plt.close()

# Main loop
def main():
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_skip = 2
    frame_count = 0
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            mean_brightness = np.mean(gray)
            scale_factor = 1.15 if mean_brightness > 100 else 1.2
            min_neighbors = 5 if mean_brightness > 100 else 3
            
            canvas = detect(gray, frame, scale_factor, min_neighbors)
            cv2.imshow('Emotion Detection', canvas)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Save emotion log
        log_path = os.path.join(output_dir, 'emotions.txt')
        with open(log_path, 'w') as f:
            for emotion, count in emotion_counts.items():
                f.write(f'{emotion}: {count}\n')
        
        # Generate and save charts
        save_emotion_charts()
        
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()