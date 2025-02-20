import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

MODEL_PATH = 'Models/emotion_model_best.keras'

def create_or_load_model():
    """
    Load an existing model if available, otherwise create a new one.
    """
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
    else:
        print("No existing model found. Creating a new model...")
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Second convolutional block
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Third convolutional block
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Flatten and dense layers
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')  # 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
        ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Create or load the model
model = create_or_load_model()

# Save the model after updating
model.save(MODEL_PATH)
print(f"Model has been saved as '{MODEL_PATH}'")