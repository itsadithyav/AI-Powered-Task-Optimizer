import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.ndimage import gaussian_filter
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time

class FrameBuffer:
    def __init__(self, maxsize=30):
        self.frame_queue = Queue(maxsize=maxsize)
        self.result_queue = Queue(maxsize=maxsize)
        self.running = True

class EmotionPredictor:
    def __init__(self, buffer_size=10):
        self.prediction_history = {}
        self.buffer_size = buffer_size
        self.face_trackers = {}
        self.next_face_id = 0
        self.lock = threading.Lock()  # Add thread safety

    def update_prediction(self, face_id, emotion_probs):
        with self.lock:  # Thread-safe updates
            if face_id not in self.prediction_history:
                self.prediction_history[face_id] = deque(maxlen=self.buffer_size)

            self.prediction_history[face_id].append(emotion_probs)

            weights = np.linspace(0.5, 1.0, len(self.prediction_history[face_id]))
            weights = weights / np.sum(weights)

            smoothed_prediction = np.zeros_like(emotion_probs)
            for i, pred in enumerate(self.prediction_history[face_id]):
                smoothed_prediction += pred * weights[i]

            return smoothed_prediction

class EmotionDetectionSystem:
    def __init__(self, model_path='emotion_model_best.h5', num_workers=4):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_model = load_model(model_path)
        self.emotion_predictor = EmotionPredictor()
        self.frame_buffer = FrameBuffer()
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.fps_counter = deque(maxlen=30)

    def process_face(self, face_data):
        """
Process a single face in a separate thread
        """
        frame, (x, y, w, h) = face_data
        padding = int(0.1 * w)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, frame.shape[1])
        y2 = min(y + h + padding, frame.shape[0])

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return None

        preprocessed_face = self.preprocess_face(face_roi)
        emotion_predictions = self.emotion_model.predict(preprocessed_face)

        current_bbox = (x, y, w, h)
        with self.emotion_predictor.lock:
            face_id = self.emotion_predictor.next_face_id
            self.emotion_predictor.next_face_id += 1


        smoothed_predictions = self.emotion_predictor.update_prediction(
            face_id,
            emotion_predictions[0]
        )

        return (face_id, current_bbox, smoothed_predictions, (x1, y1, x2, y2))

    def process_frame(self, frame):
        """
Process frame with multi-threaded face detection and emotion recognition
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            equalized,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Process faces in parallel
        face_data = [(frame, face) for face in faces]
        if face_data:
            futures = [self.executor.submit(self.process_face, data)
                       for data in face_data]
            results = [future.result() for future in futures
                       if future.result() is not None]
        else:
            results = []

        return self.draw_results(frame, results)

    def capture_frames(self):
        """
Capture frames in a separate thread
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while self.frame_buffer.running:
            ret, frame = cap.read()
            if not ret:
                break

            if not self.frame_buffer.frame_queue.full():
                self.frame_buffer.frame_queue.put(frame)

        cap.release()

    def process_frames(self):
        """
Process frames in a separate thread
        """
        while self.frame_buffer.running:
            if not self.frame_buffer.frame_queue.empty():
                frame = self.frame_buffer.frame_queue.get()
                start_time = time.time()

                processed_frame = self.process_frame(frame)

                # Calculate and display FPS
                processing_time = time.time() - start_time
                self.fps_counter.append(1.0 / processing_time)
                fps = np.mean(self.fps_counter)
                cv2.putText(processed_frame, f"FPS: {fps:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                if not self.frame_buffer.result_queue.full():
                    self.frame_buffer.result_queue.put(processed_frame)

    def display_frames(self):
        """
Display processed frames in a separate thread
        """
        while self.frame_buffer.running:
            if not self.frame_buffer.result_queue.empty():
                frame = self.frame_buffer.result_queue.get()
                cv2.imshow('Emotion Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.frame_buffer.running = False
                    break

    def draw_results(self, frame, results):
        """
Draw detection results on the frame
        """
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        current_faces = {}

        for face_id, bbox, predictions, coords in results:
            x, y, w, h = bbox
            x1, y1, x2, y2 = coords
            current_faces[face_id] = bbox

            emotion_label = emotions[np.argmax(predictions)]
            confidence = np.max(predictions)

            # Calculate stability
            if (face_id in self.emotion_predictor.prediction_history and
                len(self.emotion_predictor.prediction_history[face_id]) > 1):
                stability = self.calculate_prediction_stability(
                    self.emotion_predictor.prediction_history[face_id]
                )
            else:
                stability = 1.0

            # Draw results
            color = (0, int(255 * stability), 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{emotion_label}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            stability_label = f"Stability: {stability:.2f}"
            cv2.putText(frame, stability_label, (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self.emotion_predictor.face_trackers = current_faces
        return frame

    @staticmethod
    def calculate_prediction_stability(prediction_history):
        """
Calculate stability score based on prediction history
        """
        predictions = np.array(prediction_history)
        std_dev = np.std(predictions, axis=0)
        stability = 1.0 - np.mean(std_dev)
        return max(0.0, min(1.0, stability))

    @staticmethod
    def preprocess_face(face_img, target_size=(48, 48)):
        """
Preprocess face image for emotion detection
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        denoised = cv2.bilateralFilter(equalized, d=9, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        smoothed = gaussian_filter(enhanced, sigma=0.5)
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
    """
Main function to run the multi-threaded emotion detection system
    """
    # Initialize the system
    system = EmotionDetectionSystem(num_workers=4)

    # Create threads for capture, processing, and display
    capture_thread = threading.Thread(target=system.capture_frames)
    process_thread = threading.Thread(target=system.process_frames)
    display_thread = threading.Thread(target=system.display_frames)

    # Start threads
    print("Starting emotion detection system...")
    capture_thread.start()
    process_thread.start()
    display_thread.start()

    # Wait for threads to complete
    capture_thread.join()
    process_thread.join()
    display_thread.join()

    # Cleanup
    cv2.destroyAllWindows()
    system.executor.shutdown()

if __name__ == "__main__":
    main()