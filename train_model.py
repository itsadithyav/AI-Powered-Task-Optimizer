import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import os
import datetime
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.model_selection import train_test_split

# Enable mixed precision (Speeds up training on modern GPUs)
set_global_policy('float32') 
tf.keras.mixed_precision.set_global_policy('mixed_float16')
#tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()  # Debugging data pipeline issues


MODEL_PATH = 'Models/emotion_model_best.keras'

# Ensure GPU memory is used efficiently
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def download_fer2013():
    """Download FER2013 dataset if not already present."""
    dataset_path = 'data/fer2013.csv'
    if not os.path.exists(dataset_path):
        os.makedirs('data', exist_ok=True)
        url = "https://www.dropbox.com/s/l6g1k8c3l6f9yzo/fer2013.csv?dl=1"
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(dataset_path, 'wb') as f:
            for data in tqdm(response.iter_content(1024), total=total_size//1024, desc="Downloading FER2013"):
                f.write(data)
        print("Dataset downloaded successfully!")
    else:
        print("Dataset already exists. Skipping download.")

def prepare_data():
    """Load and preprocess FER2013 dataset."""
    try:
        data = pd.read_csv('data/fer2013.csv')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None, None, None

    # Convert pixel values properly
    X = np.array([np.array(pixels.split(), dtype=np.float32).reshape(48, 48, 1) / 255.0 for pixels in data['pixels']])
    y = to_categorical(data['emotion'], num_classes=7)
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def create_model():
    """Define an optimized CNN model for emotion detection."""
    inputs = tf.keras.Input(shape=(48, 48, 1))

    def residual_block(x, filters):
        """Residual Block for deeper feature extraction."""
        shortcut = x
        x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([shortcut, x])  # Skip Connection
        return x

    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    x = tf.keras.layers.Dropout(0.2)(x)

    x = residual_block(x, 64)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    x = tf.keras.layers.Dropout(0.2)(x)

    x = residual_block(x, 128)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(7, activation='softmax', dtype='float32')(x)  # Use float32 to avoid precision issues

    model = tf.keras.Model(inputs, outputs)

    # AdamW optimizer with weight decay
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train or retrain the emotion detection model."""
    print("Downloading dataset...")
    download_fer2013()
    
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    
    if X_train is None:
        print("Error in preparing data. Exiting.")
        return
    
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Loaded existing model for retraining.")

        # ✅ Recompile with a fresh optimizer to avoid variable mismatch issues
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        print("No existing model found. Creating a new model.")
        model = create_model()


    # Callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint = ModelCheckpoint(
        f"checkpoints/emotion_model_{timestamp}.weights.h5",
        monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # Optimized Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Convert dataset to efficient TensorFlow Dataset pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64).shuffle(1000).prefetch(AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(AUTOTUNE)

    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=100,
        validation_data=test_dataset,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    model.save(MODEL_PATH)
    print(f"Training completed. Model saved as '{MODEL_PATH}'")
    
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    return history

if __name__ == "__main__":
    history = train_model()
