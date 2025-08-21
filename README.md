# Rayan's Emotions Detector

A real-time facial feature and emotion detection system that analyzes webcam video feeds to detect faces, eyes, smiles, and classify emotions using computer vision and deep learning techniques.

## 🎯 Features

- **Real-time Face Detection**: Identifies faces using OpenCV's Haar cascade classifiers
- **Facial Feature Detection**: Detects eyes and smiles with confidence scoring
- **Emotion Recognition**: Classifies 7 emotions (Happy, Sad, Neutral, Angry, Fear, Surprise, Disgust) using DeepFace
- **Live Visualization**: Displays annotated video feed with detection results and FPS
- **Data Logging**: Records emotion counts in a text file
- **Chart Generation**: Creates bar and pie charts showing emotion distribution
- **Dynamic Optimization**: Adapts detection parameters based on lighting conditions

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **OpenCV** - Computer vision and image processing
- **DeepFace** - Deep learning-based emotion recognition
- **TensorFlow** - Neural network backend
- **Matplotlib** - Data visualization
- **NumPy** - Numerical computations

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam
- Sufficient lighting for better detection accuracy

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd "Rayan's Emotions Detector"
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

1. **Run the emotion detector**

   ```bash
   python self_done_homework.py
   ```

2. **Controls**

   - Press `q` to quit the application
   - The program will automatically save results when you exit

3. **Output Files**
   After running the program, check the `output/` folder for:
   - `emotions.txt` - Emotion count log
   - `emotion_bar_chart.png` - Bar chart visualization
   - `emotion_pie_chart.png` - Pie chart visualization

## 📁 Project Structure

```
Rayan's Emotions Detector/
├── self_done_homework.py          # Main application file
├── requirements.txt               # Python dependencies
├── README.md                     # Project documentation
├── haarcascade_frontalface_default.xml  # Face detection model
├── haarcascade_eye.xml           # Eye detection model
├── haarcascade_smile.xml         # Smile detection model
├── output/                       # Generated output files
│   ├── emotions.txt              # Emotion count log
│   ├── emotion_bar_chart.png     # Bar chart
│   └── emotion_pie_chart.png     # Pie chart
└── venv/                         # Virtual environment (not in repo)
```

## 🔧 How It Works

1. **Video Capture**: Captures live video from your webcam
2. **Face Detection**: Uses Haar cascade classifiers to detect faces in each frame
3. **Feature Detection**: Identifies eyes and smiles within detected faces
4. **Emotion Analysis**: Uses DeepFace to classify emotions from facial expressions
5. **Real-time Display**: Shows annotated video with detection results
6. **Data Collection**: Logs emotion counts throughout the session
7. **Visualization**: Generates charts showing emotion distribution

## ⚙️ Configuration

The system automatically adjusts detection parameters based on lighting:

- **Bright conditions**: More conservative detection (higher confidence thresholds)
- **Low light**: More sensitive detection (lower confidence thresholds)

## 🐛 Troubleshooting

### Common Issues

1. **Webcam not detected**

   - Ensure your webcam is connected and not being used by another application
   - Try changing the camera index in the code (currently set to 0)

2. **Poor detection accuracy**

   - Ensure good lighting conditions
   - Position yourself clearly in front of the camera
   - Keep your face at a reasonable distance from the camera

3. **Performance issues**

   - The program uses frame skipping to improve performance
   - Close other resource-intensive applications
   - Ensure you have sufficient RAM and processing power

4. **Import errors**
   - Make sure you've activated the virtual environment
   - Verify all dependencies are installed: `pip install -r requirements.txt`

### Error Messages

- **"Failed to load cascade classifier"**: Ensure all XML files are in the project directory
- **"No module named 'deepface'"**: Install dependencies with `pip install -r requirements.txt`
- **"Camera not accessible"**: Check if webcam is available and not in use by another application

## 📊 Output Interpretation

### Emotion Categories

- **Happy**: Positive emotions, smiles
- **Sad**: Negative emotions, frowns
- **Neutral**: Balanced facial expression
- **Angry**: Aggressive facial features
- **Fear**: Anxious or scared expressions
- **Surprise**: Shocked or amazed expressions
- **Disgust**: Repulsed or offended expressions

### Charts

- **Bar Chart**: Shows count of each detected emotion
- **Pie Chart**: Shows percentage distribution of emotions

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Rayan** - Developer of this emotion detection system

---

**Note**: This system is designed for educational and research purposes. Ensure you have proper consent when using it to analyze other people's emotions.
