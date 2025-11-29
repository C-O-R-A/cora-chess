# '''
# Convert the board fens and moves to tensors for CNN training
# then train a CNN model to output probability distribution over all possible moves
# then save the model for deployment
# '''

# import chess
# import torch 
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from datetime import datetime


# def construct_piece_layer(board_fen, piece):
#     '''
#     Creates feature map for each piece
#     '''
#     board = chess.Board(board_fen)
#     board_str = str(board)
#     board_str = board_str.replace(' ', '').replace('\n', '')
#     arr = np.zeros((8,8), dtype=np.int32)
#     for i,char in enumerate(board_str):
#         row, col = divmod(i, 8)
#         if char == piece:
#             arr[row, col] = -1
#         elif char == piece.upper():
#             arr[row, col] = 1

#     return arr

# def convert_board_to_tensor(board):
#     '''
#     Convert the chess board to a tensor representation
#     channels: 6 (6 pieces
#     shape: 8x8x6
#     black as -1, white as 1
#     '''
#     pieces = ['p', 'r', 'n', 'b', 'q', 'k']
#     layers = []
#     for piece in pieces:
#         piece_layer = construct_piece_layer(board, piece)
#         layers.append(piece_layer)
#     tensor = np.stack(layers)
#     return tensor

# def process_eco_code(eco, eco_to_idx):
#     '''
#     Encodes ECO code to an integer index.
#     '''
#     return eco_to_idx.get(eco, 0)  # default to 0 if not found

# def process_move_number(move_no):
#     return int(move_no)

# def process_date_played(date_played):
#     date_played = date_played.replace("??", "01")
#     dt = datetime.strptime(date_played, "%Y.%m.%d")

#     epoch = datetime(1970, 1, 1)
#     delta_days = (dt - epoch).days
#     return delta_days

# def process_color(color):
#     return 0 if color == 'white' else 1

# def process_count(count):
#     return int(count)

# def expand_tensor(board_tensor, extra_data, legal_moves):
#     '''
#     board_tensor: (6, 8, 8) array \n
#     extra_data: list of extra features \n
#     legal_moves: list of move tuples [(4, 6), (2, 7), ...]  \n
#     returns: dict with both
#     '''
#     extra_tensor = torch.tensor(extra_data, dtype=torch.float32) # CNN's expect float64
#     board_tensor = torch.tensor(board_tensor, dtype=torch.float32)
#     legal_moves_tensor = torch.tensor(legal_moves, dtype=torch.float32)
    
#     return {
#         'board': board_tensor,
#         'extra': extra_tensor,
#         'legal_moves': legal_moves_tensor
#     }

# def convert_move_to_ranking(legal_moves, target_move):
#     for i, move in enumerate(legal_moves):
#         if move == target_move:
#             return i   # return index of correct move
#     return -1


# def convert_move_to_tensor(move):
#     '''
#     Convert a chess move string to a tensor representation
#     e.g. e2e4 -> [4, 1] (from square, to square)

#     input: move in UCI format (e.g. e2e4) (string)
#     output: returns from and to tensor
#     '''
#     move_obj = chess.Move.from_uci(move)
#     from_square = move_obj.from_square
#     to_square = move_obj.to_square

#     from_layer = np.zeros((8,8), dtype=float)
#     to_layer = np.zeros((8,8), dtype=float)
    
#     from_row, from_col = divmod(from_square, 8)
#     to_row, to_col = divmod(to_square, 8)

#     from_layer[from_row, from_col] = 1
#     to_layer[to_row, to_col] = 1
    
#     return from_layer, to_layer


