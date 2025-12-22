import numpy as np
import utils
import torch.nn as nn
import chess
import torch

class Agent():
    def __init__(self, model):
        self.model = model
        return
    
    def choose_move(self, board, extra):
        legal_moves = [(m.from_square // 8, m.from_square % 8, m.to_square // 8, m.to_square % 8) for m in board.legal_moves]
        board_tensor = utils.convert_board_to_tensor(board)

        scores = self.model(board_tensor, extra, legal_moves)

        target_move = scores[max(scores)]
        from_row, from_col, to_row, to_col = target_move
        from_square = from_row * 8 + from_col
        to_square = to_row * 8 + to_col
        idx = from_square * 64 + to_square

        return board.legal_moves
    
class Board():
    def __init__(self):
        self.position = np.zeros(3)  # x,y,z
        self.rotation = np.zeros(3)  # rx,ry,rz
        self.cell_size = 0.02 # 2cm default
        self.Board = chess.Board()
        self.transformation_matrix = np.eye(4)
        return
    
    def update_board_state(self, state_fen): ## should be called once every other turn right before the robot plans its move
        self.Board.set_fen(state_fen)
        return
    
    def update_board_pose(self, pose_vector): ## should be called continuously to keep track of board position
        self.position = pose_vector[0:3]
        self.rotation = pose_vector[3:6]
        return
    
    def get_board_state(self):
        return self.Board.fen()
    
