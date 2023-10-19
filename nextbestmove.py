import chess
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('2500elo_predictor.h5')

# Define mapping for each chess piece to its integer representation
mapping = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
}


# Convert FEN to board tensor
def fen_to_board_tensor(fen):
    board = chess.Board(fen)
    tensor = [[0 for _ in range(8)] for _ in range(8)]
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            tensor[7 - square // 8][square % 8] = mapping[piece.symbol()]
    return tensor

# Define the function to get the best move
def get_best_move(board):
    best_move = None
    best_delta = 0  # Initializing to 0

    is_white_turn = board.turn == chess.WHITE

    for move in board.legal_moves:
        print(move)

        board.push(move)
        board_tensor = fen_to_board_tensor(board.fen())
        board_tensor_reshaped = np.array(board_tensor).reshape(1, 8, 8, 1)

        predicted_delta = model.predict(board_tensor_reshaped)[0][0]
        print("Predicted Delta:", predicted_delta)

        if is_white_turn:
            if best_move is None or predicted_delta > best_delta:  
                best_delta = predicted_delta
                best_move = move

        else:
            if best_move is None or predicted_delta < best_delta:  
                best_delta = predicted_delta
                best_move = move

        board.pop()

    return best_move

# Test the function
fen_position = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
board = chess.Board(fen_position)
best_move = get_best_move(board)
print(f"Best Move: {best_move}")
