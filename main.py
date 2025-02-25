import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.ndimage import gaussian_filter
import time
import hashlib

# User authentication system
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    try:
        with open("users.txt", "r") as file:
            users = dict(line.strip().split(":") for line in file)
        return users
    except FileNotFoundError:
        return {}

def save_user(username, password):
    with open("users.txt", "a") as file:
        file.write(f"{username}:{hash_password(password)}\n")

def authenticate(username, password):
    users = load_users()
    return users.get(username) == hash_password(password)

class EmotionDetectionSystem:
    def __init__(self, model_path='Models/emotion_model_best.keras'):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = load_model(model_path)

    def process_face(self, face_data):
        frame, (x, y, w, h) = face_data
        padding = int(0.1 * w)
        x1, y1 = max(x - padding, 0), max(y - padding, 0)
        x2, y2 = min(x + w + padding, frame.shape[1]), min(y + h + padding, frame.shape[0])

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return None, None

        preprocessed_face = self.preprocess_face(face_roi)
        emotion_predictions = self.emotion_model.predict(preprocessed_face)

        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_label = emotions[np.argmax(emotion_predictions)]
        confidence = np.max(emotion_predictions)

        return emotion_label, confidence

    def capture_and_process_frames(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(equalized, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

            detected_mood = "Detecting..."
            assigned_task = "No task assigned."
            for (x, y, w, h) in faces:
                emotion_label, confidence = self.process_face((frame, (x, y, w, h)))

                if emotion_label:
                    detected_mood = f"{emotion_label} ({confidence:.2f})"
                    assigned_task = assign_task_based_on_mood(emotion_label)
                
                color = (0, 255, 0)  # Color for bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{emotion_label}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield frame_bytes, detected_mood, assigned_task

        cap.release()

    @staticmethod
    def preprocess_face(face_img, target_size=(48, 48)):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        smoothed = gaussian_filter(equalized, sigma=0.5)
        resized = cv2.resize(smoothed, target_size, interpolation=cv2.INTER_CUBIC)
        normalized = resized.astype('float32') / 255.0
        gamma_corrected = np.power(normalized, 1.2)
        min_val = np.min(gamma_corrected)
        max_val = np.max(gamma_corrected)
        stretched = (gamma_corrected - min_val) / (max_val - min_val)
        preprocessed = img_to_array(stretched)
        preprocessed = np.expand_dims(preprocessed, axis=0)
        return preprocessed

def load_tasks_from_file(filename="tasks.txt"):
    try:
        with open(filename, "r") as file:
            tasks = file.readlines()
        return {line.split(':')[0].strip(): line.split(':')[1].strip() for line in tasks if ':' in line}
    except FileNotFoundError:
        return {}

def assign_task_based_on_mood(mood):
    task_dict = load_tasks_from_file()
    return task_dict.get(mood, "No specific task assigned.")

def main():
    st.title("Real-time Emotion Detection and Task Manager")
    
    users = load_users()
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        choice = st.sidebar.selectbox("Login / Register", ["Login", "Register"])
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if choice == "Register":
            if st.sidebar.button("Register"):
                if username and password:
                    if username in users:
                        st.sidebar.warning("Username already exists!")
                    else:
                        save_user(username, password)
                        st.sidebar.success("User registered successfully!")
                else:
                    st.sidebar.warning("Please enter a username and password.")
        else:
            if st.sidebar.button("Login"):
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.sidebar.success("Login successful!")
                else:
                    st.sidebar.error("Invalid username or password.")
        return

    system = EmotionDetectionSystem(model_path='Models/emotion_model_best.keras')

    video_placeholder = st.empty()
    detected_mood_placeholder = st.empty()
    assigned_task_placeholder = st.empty()

    for frame_bytes, mood, assigned_task in system.capture_and_process_frames():
        video_placeholder.image(frame_bytes, channels="BGR", use_container_width=True)
        detected_mood_placeholder.text(f"Detected Mood: {mood}")
        assigned_task_placeholder.text(f"Assigned Task: {assigned_task}")

if __name__ == "__main__":
    main()
