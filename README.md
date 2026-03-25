# AI-Powered Task Optimizer

## Description
A machine learning-based system designed to analyze employee emotions and recommend tasks aligned with their mood. By leveraging deep learning models, the system dynamically analyzes facial expressions, text, and audio inputs to enhance productivity, improve well-being, and detect signs of stress or burnout for timely intervention.

## Key Features
- **Real-Time Emotion Detection**: Captures and processes live video feeds to determine the user's emotional state using OpenCV and TensorFlow.
- **Multi-Modal Deep Learning**: Employs a robust model utilizing facial features (CNNs), text configurations, and audio elements (LSTMs).
- **Intelligent Task Assignment**: Recommends appropriate tasks based on the user's prevailing mood to maximize operational efficiency.
- **Secure User Authentication**: Features a built-in login and registration system utilizing SHA-256 password hashing to ensure data privacy.
- **Interactive Web Interface**: Provides an intuitive, easy-to-use application dashboard powered by Streamlit.

## Tech Stack
- **Core**: Python
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Web UI**: Streamlit
- **Data Engineering**: NumPy, Pandas, SciPy, Scikit-learn

## How It Works
1. **User Authentication**: A user securely logs into the Streamlit dashboard using an encrypted login mechanism.
2. **Data Capture**: The system securely captures the live video feed through the user's webcam.
3. **Face Detection & Preprocessing**: Face regions are extracted using Haar Cascades, equalized, smoothed, and normalized.
4. **Emotion Prediction**: Preprocessed facial data is passed into the pre-trained deep learning classification model, generating confidence scores across 7 emotion categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
5. **Task Assignment**: A mapping engine links the predicted emotion label to a customized task list, assigning appropriate tasks based on the user's current cognitive state.

## Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/itsadithyav/AI-Powered-Task-Optimizer.git
   cd AI-Powered-Task-Optimizer
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy pandas tensorflow scipy scikit-learn streamlit
   ```

3. **Data & Weights Configuration**:
   - Ensure a mapping file exists at `data/tasks.txt` for task assignments.
   - The pre-trained model should be situated at `Models/emotion_model_best.keras` or `Models/multi_input_mood_model.keras`. Create a new model configuration using `python create_model.py`.
   - Optionally run `python train_model.py` to retrain the architecture on updated datasets in the `data/` directory.

## Usage
1. **Launch the Application**:
   Run the interactive web UI using Streamlit:
   ```bash
   streamlit run main.py
   ```
2. **Authenticate**: Register a new user account or log in with existing credentials on the left sidebar.
3. **Active Detection**: Allow camera permissions. The dashboard will dynamically process frames, overlaying the detected mood and prescribing the active task.

## Example
- **Input**: Live webcam feed capturing a smiling user
- **Detection Result**: `Happy (0.94)`
- **Assigned Output**: The Streamlit interface displays the assigned task mapped to "Happy" in the `tasks.txt` schema (e.g., *Creative brainstorming session*).

## Future Improvements
- **Scalable Cloud Deployment**: Containerize the solution utilizing Docker and host it on providers like AWS or GCP for reliable load scaling.
- **Advanced Fusion Algorithms**: Enhance the multi-modal integration between real-time video, NLP, and speech recognition to elevate classification coherence.
- **HR Dashboard & Analytics Alerting**: Integrate automated notifications for HR personnel when critical long-term stress signals are flagged.

## License
This project is licensed under the [MIT License](LICENSE).
