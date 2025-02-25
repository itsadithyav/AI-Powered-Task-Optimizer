import tensorflow as tf
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
                                     LSTM, Embedding, TimeDistributed, GlobalAveragePooling1D, Concatenate)
from tensorflow.keras.optimizers import Adam

MODEL_PATH = 'Models/multi_input_mood_model.keras'

# Define input shapes
IMAGE_SHAPE = (48, 48, 1)  # Facial data (grayscale images)
TEXT_MAXLEN = 100  # Maximum text sequence length
TEXT_VOCAB_SIZE = 20000  # Vocabulary size for text
TEXT_EMBEDDING_DIM = 128  # Embedding dimension for text
AUDIO_SHAPE = (50, 13)  # Example shape for audio (Mel spectrogram features)


def create_or_load_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
    else:
        print("No existing model found. Creating a new model...")
        
        # Facial input
        image_input = Input(shape=IMAGE_SHAPE, name='image_input')
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        image_features = Dense(128, activation='relu')(x)
        
        # Text input
        text_input = Input(shape=(TEXT_MAXLEN,), name='text_input')
        text_embedding = Embedding(TEXT_VOCAB_SIZE, TEXT_EMBEDDING_DIM)(text_input)
        text_lstm = LSTM(128)(text_embedding)
        text_features = Dense(128, activation='relu')(text_lstm)
        
        # Audio input
        audio_input = Input(shape=AUDIO_SHAPE, name='audio_input')
        audio_lstm = LSTM(128, return_sequences=True)(audio_input)
        audio_pooled = GlobalAveragePooling1D()(audio_lstm)
        audio_features = Dense(128, activation='relu')(audio_pooled)
        
        # Combine all features
        combined_features = Concatenate()([image_features, text_features, audio_features])
        x = Dense(256, activation='relu')(combined_features)
        x = Dropout(0.5)(x)
        output = Dense(7, activation='softmax', name='output')  # 7 mood classes
        
        model = Model(inputs=[image_input, text_input, audio_input], outputs=output)
        
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
