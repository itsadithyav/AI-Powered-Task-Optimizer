import numpy as np
import pandas as pd
import tensorflow as tf
import os
import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.model_selection import train_test_split

# Enable mixed precision (Speeds up training on modern GPUs)
set_global_policy('float32')
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.data.experimental.enable_debug_mode()

MODEL_PATH = 'Models/emotion_model_best.keras'

def prepare_data():
    """Load and preprocess datasets for text, audio, and facial recognition."""
    try:
        face_data = pd.read_csv('data/facial_data.csv')
        text_data = pd.read_csv('data/text_data.csv')
        audio_data = pd.read_csv('data/audio_data.csv')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None, None, None, None, None

    # Facial data preprocessing
    X_face = np.array([np.array(pixels.split(), dtype=np.float32).reshape(48, 48, 1) / 255.0 for pixels in face_data['pixels']])
    y = to_categorical(face_data['emotion'], num_classes=7)
    
    # Text data preprocessing
    X_text = np.array(text_data['features'].tolist())  # Assuming text features are already extracted
    
    # Audio data preprocessing
    X_audio = np.array(audio_data['features'].tolist())  # Assuming audio features are already extracted
    
    return train_test_split(X_face, X_text, X_audio, y, test_size=0.2, random_state=42, stratify=y)

def create_model():
    """Define a multi-input model for emotion detection."""
    # Facial recognition input
    face_input = tf.keras.Input(shape=(48, 48, 1), name='face_input')
    x_face = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(face_input)
    x_face = tf.keras.layers.MaxPooling2D((2, 2))(x_face)
    x_face = tf.keras.layers.Flatten()(x_face)
    
    # Text input
    text_input = tf.keras.Input(shape=(100,), name='text_input')  # Assuming 100-dimensional text features
    x_text = tf.keras.layers.Dense(128, activation='relu')(text_input)
    
    # Audio input
    audio_input = tf.keras.Input(shape=(100,), name='audio_input')  # Assuming 100-dimensional audio features
    x_audio = tf.keras.layers.Dense(128, activation='relu')(audio_input)
    
    # Concatenate features from all three modalities
    merged = tf.keras.layers.concatenate([x_face, x_text, x_audio])
    merged = tf.keras.layers.Dense(512, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(0.5)(merged)
    outputs = tf.keras.layers.Dense(7, activation='softmax', dtype='float32')(merged)
    
    model = tf.keras.Model(inputs=[face_input, text_input, audio_input], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-5),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    return model

def train_model():
    """Train or retrain the multi-input emotion detection model."""
    print("Preparing data...")
    X_face_train, X_face_test, X_text_train, X_text_test, X_audio_train, X_audio_test, y_train, y_test = prepare_data()
    
    if X_face_train is None:
        print("Error in preparing data. Exiting.")
        return
    
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Loaded existing model for retraining.")
    else:
        print("No existing model found. Creating a new model.")
        model = create_model()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint = ModelCheckpoint(f"checkpoints/emotion_model_{timestamp}.weights.h5", monitor='val_accuracy',
                                 mode='max', save_best_only=True, save_weights_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=15, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
    
    print("Training model...")
    history = model.fit(
        [X_face_train, X_text_train, X_audio_train], y_train,
        batch_size=128,
        epochs=150,
        validation_data=([X_face_test, X_text_test, X_audio_test], y_test),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    model.save(MODEL_PATH)
    print(f"Training completed. Model saved as '{MODEL_PATH}'")
    
    test_loss, test_accuracy = model.evaluate([X_face_test, X_text_test, X_audio_test], y_test)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    return history

if __name__ == "__main__":
    history = train_model()
