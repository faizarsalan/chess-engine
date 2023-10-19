import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# === Dataset Preparation ===

def load_pgn(pgn_path):
    with open(pgn_path, 'r') as f:
        return [game for game in chess.pgn.read_game(f)]

def extract_data_from_games(games):
    data = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            fen = board.fen()
            data.append((fen, move))
            board.push(move)
    return data

# Load games and extract data
games = load_pgn('2500elo.pgn')
data = extract_data_from_games(games)

# === Feature Engineering ===

def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = [[0 for _ in range(8)] for _ in range(8)]
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            tensor[7 - square // 8][square % 8] = mapping[piece.symbol()]
    return tensor

def move_to_delta(move):
    return (move.to_square - move.from_square)

# mapping = {symbol: i+1 for i, symbol in enumerate('PNBRQKpnbrqk')} --> 2nd alternative
mapping = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
}
X = [fen_to_tensor(fen) for fen, _ in data]
y = [move_to_delta(move) for _, move in data]

# Splitting the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# === Model Architecture and Training ===

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(8, 8, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

X_train_reshaped = np.array(X_train).reshape(len(X_train), 8, 8, 1)
X_val_reshaped = np.array(X_val).reshape(len(X_val), 8, 8, 1)

y_train_np = np.array(y_train)
y_val_np = np.array(y_val)

# model.fit(X_train_reshaped, y_train_np, epochs=100, batch_size=,validation_data=(X_val_reshaped, y_val_np))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('2500elo_predictor.h5', monitor='val_loss', save_best_only=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_delta=0.0001)

# callbacks_list = [early_stopping, model_checkpoint, reduce_lr]

# Fitting the model with the additional parameters
model.fit(
    X_train_reshaped, 
    y_train_np, 
    epochs=100,  # Increased epochs
    batch_size=128,  # Adjust the batch size
    validation_data=(X_val_reshaped, y_val_np),
    shuffle=True,
    callbacks=model_checkpoint
)

import matplotlib.pyplot as plt

# Training the model and storing its history
history = model.fit(
    X_train_reshaped, 
    y_train_np, 
    epochs=100,  # Increased epochs
    batch_size=128,  # Adjust the batch size
    validation_data=(X_val_reshaped, y_val_np),
    shuffle=True,
    callbacks=model_checkpoint
)

# Plotting training and validation loss
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# === Move Prediction ===

def predict_move(fen):
    tensor = fen_to_tensor(fen)
    tensor_reshaped = np.array(tensor).reshape(1, 8, 8, 1)
    delta = model.predict(tensor_reshaped)[0][0]
    return delta

# Testing move prediction
test_fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
predicted_delta = predict_move(test_fen)
print(f"Predicted Move Delta: {predicted_delta}")

# === Save the model ===
model.save('2500elo_predictor.h5')
