import chess
import chess.pgn

def parsePGN(filepath):
    data = []
    labels = []

    with open(filepath) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            board = game.board()
            result = game.headers["Result"]
            
            if result == "1-0":
                labels.append(1)
            elif result == "0-1":
                labels.append(-1)
            else:
                labels.append(0)

            moves = list(game.mainline_moves())
            game_data = []

            for move in moves:
                board.push(move)
                fen = board.fen()
                game_data.append(fen)

            data.append(game_data)
            
    return data, labels


def main():
    print("Hello World")


if __name__ == "__main__":
    main()
    