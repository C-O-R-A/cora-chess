import numpy as np
from datetime import datetime
import torch

def convert_board_to_tensor(chessboard):
    '''
    Docstring for convert_board_to_tensor

    :param chessboard: chessboard as chess.Board object
    '''
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    board = chessboard.fen()
    board_str = str(board).replace(' ','').replace('\n','')
    layers = []
    for piece in pieces:
        arr = np.zeros((8,8), dtype=np.float32)
        for i,char in enumerate(board_str):
            row, col = divmod(i, 8)
            if char == piece:
                arr[row, col] = -1
            elif char == piece.upper():
                arr[row, col] = 1
        layers.append(arr)

    return torch.tensor(np.stack(layers), dtype=torch.float32).unsqueeze(0)

def generate_legal_moves(legal_moves):
    '''
    Docstring for generate_legal_moves
    
    :param legal_moves: list of legal moves as chess.Move objects
    '''
    legal_moves = [(m.from_square // 8, m.from_square % 8, m.to_square // 8, m.to_square % 8) for m in legal_moves]
    legal_moves_tensor = torch.tensor(legal_moves, dtype=torch.float32)  # [N,4]
    return legal_moves_tensor

def process_date_played(date_played):
    date_played = date_played.replace("??", "01")
    dt = datetime.strptime(date_played, "%Y.%m.%d")
    epoch = datetime(1970,1,1)
    delta_days = (dt - epoch).days
    return delta_days