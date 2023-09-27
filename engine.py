import numpy as np
import chess
import chess.svg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set random seed for TensorFlow (model initialization and training)
tf.random.set_seed(42)

# Define the neural network model for feature extraction
def create_feature_extraction_model():
    model = keras.Sequential([
        layers.Input(shape=(8, 8, 12)),  # 8x8 chess board with 12 channels (one-hot encoding)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(2, activation='relu'),
        layers.Dense(5, activation='tanh') # multiple features representation
        # layers.Dense(1, activation='tanh')  # Output layer for feature representation
    ])
    return model

# Sample FEN position (replace with your data loading logic)
sample_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Convert FEN position to one-hot encoding
def fen_to_one_hot(fen):
    board = chess.Board(fen)
    board_state = np.zeros((8, 8, 12), dtype=np.float32)  # Initialize an empty board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_idx = piece.piece_type - 1 + 6 * piece.color
            row = square // 8
            col = square % 8
            board_state[row, col, piece_idx] = 1
    return board_state

# Create the feature extraction model
feature_extraction_model = create_feature_extraction_model()

# Convert the sample FEN to one-hot encoding
input_data = fen_to_one_hot(sample_fen)

# Reshape the input data to match the model's input shape
input_data = input_data.reshape(1, 8, 8, 12)

# Extract features from the FEN position
features = feature_extraction_model.predict(input_data)

# Print the extracted features
print("Extracted Features:", features)