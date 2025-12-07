import chess
import torch
import time
from itertools import zip_longest
import matplotlib.pyplot as plt
import random

# Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet().to(device)
model.load_state_dict(torch.load('models/magnus_120_epochs.pth', map_location=device))
model.eval()

def print_board(board):
    print(board)
    print()

# visualize_model_weights_and_graph('models/magnus_model_weights.pth')
# sample_predict_and_plot(model, device)


# print(magnus_data[0])

# Human vs Model loop
def you_vs_robot(model):
    '''
    You play against the NN
    '''
    
    # Start a new chess game
    board = chess.Board()

    while not board.is_game_over():
        print_board(board)
        print(board.fen())

        if board.turn == chess.BLACK:
            # Human plays as white
            move_input = input("Your move (in UCI format e.g. e2e4): ")
            try:
                move = chess.Move.from_uci(move_input)
                if move in board.legal_moves:
                    board.push(move)
                    time.sleep(0.5)
                else:
                    print("Illegal move. Try again.")
            except:
                print("Invalid input. Try again.")
        else:
            # Model plays as black
            board_fen = board.fen()
            turn+=1
            extra_features = [18000, turn, 0]  # or calculate dynamically if you have that logic
            move_uci = predict_move(model, board_fen, extra_features, device)
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                print(f"Model plays: {move}")
                board.push(move)
            else:
                print("Model tried an illegal move. Picking random legal move.")
                board.push(list(board.legal_moves)[0])

def robot_vs_robot(model):
    iterations = 100
    total_white_illegal_moves = []
    total_black_illegal_moves = []

    for i in range(iterations):
        print('iteration', i)
        white_illegal_moves = []
        black_illegal_moves = []
        turn = 0
        day = random.randint(1, 12000)  # returns an integer between 1 and 10 (inclusive)
        
        # Start a new chess game
        board = chess.Board()        

        # Get list of legal moves
        legal_moves = list(board.legal_moves)

        # Pick one at random
        random_move = random.choice(legal_moves)

        # Push it onto the board
        board.push(random_move)

        while not board.is_game_over():
            print_board(board)
            print(board.fen())

            if board.turn == chess.WHITE:
                # Model plays as black
                board_fen = board.fen()
                extra_features = [day, turn, 1]  # or calculate dynamically if you have that logic
                move_uci = predict_move(model, board_fen, extra_features, device)
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        print(f"Model plays: {move}")
                        board.push(move)
                        white_illegal_moves.append(0)
                    else:
                        print("Model tried an illegal move. Picking random legal move.")
                        board.push(list(board.legal_moves)[0])
                        white_illegal_moves.append(1)
                except:
                    print(f"Invalid move: {move_uci}. Choosing random move.")
                    board.push(list(board.legal_moves)[0])
                    white_illegal_moves.append(1)

            else:
                # Model plays as black
                board_fen = board.fen()
                turn+=1
                extra_features = [day, turn, 0]  # or calculate dynamically if you have that logic
                move_uci = predict_move(model, board_fen, extra_features, device)
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:                   
                        print(f"Model plays: {move}")
                        board.push(move)
                        black_illegal_moves.append(0)
                    else:
                        print("Model tried an illegal move. Picking random legal move.")
                        board.push(list(board.legal_moves)[0])
                        black_illegal_moves.append(1)
                except:
                    print(f"Invalid move: {move_uci}. Choosing random move.")
                    board.push(list(board.legal_moves)[0])
                    black_illegal_moves.append(1)

        # Print result
        print_board(board)
        result = board.result()
        print(f"Game Over. Result: {result}")
        total_white_illegal_moves = sum_turnwise(total_white_illegal_moves, white_illegal_moves)
        total_black_illegal_moves = sum_turnwise(total_black_illegal_moves, black_illegal_moves)
        time.sleep(0.5)

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(total_black_illegal_moves)), total_black_illegal_moves)
    plt.title(f'Total illegal moves by black player over {iterations} games')
    plt.xlabel('turn')
    plt.ylabel('total_black_illegal_moves')
    plt.savefig("figures/magnus_120/total_black_illegal_moves.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(total_white_illegal_moves)), total_white_illegal_moves)
    plt.title(f'Total illegal moves by white player over {iterations} games')
    plt.xlabel('turn')
    plt.ylabel('total_white_illegal_moves')
    plt.savefig("figures/magnus_120/total_white_illegal_moves.png")
    plt.show()


if __name__ == '__main__':
    robot_vs_robot(model)