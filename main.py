import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.ndimage import gaussian_filter
import time

class EmotionDetectionSystem:
    def __init__(self, model_path='emotion_model_best.h5'):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = load_model(model_path)

    def process_face(self, face_data):
        frame, (x, y, w, h) = face_data
        padding = int(0.1 * w)
        x1, y1 = max(x - padding, 0), max(y - padding, 0)
        x2, y2 = min(x + w + padding, frame.shape[1]), min(y + h + padding, frame.shape[0])

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return None

        preprocessed_face = self.preprocess_face(face_roi)
        emotion_predictions = self.emotion_model.predict(preprocessed_face)

        emotion_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][np.argmax(emotion_predictions)]
        confidence = np.max(emotion_predictions)

        return emotion_label, confidence, (x1, y1, x2, y2)

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

            for (x, y, w, h) in faces:
                emotion_label, confidence, (x1, y1, x2, y2) = self.process_face((frame, (x, y, w, h)))

                color = (0, 255, 0)  # Color for bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{emotion_label}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield frame_bytes

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

def main():
    st.title("Real-time Emotion Detection")
    st.write("This is a real-time emotion detection system using your webcam.")

    system = EmotionDetectionSystem(model_path='emotion_model_best.h5')

    # Streamlit Video placeholder
    video_placeholder = st.empty()

    # Start capturing frames
    for frame_bytes in system.capture_and_process_frames():
        video_placeholder.image(frame_bytes, channels="BGR", use_container_width=True)

if __name__ == "__main__":
    main()
