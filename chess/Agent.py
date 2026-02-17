import numpy as np
import utils
import torch.nn as nn

class Agent:
    def __init__(self, model):
        self.model = model
        return

    def choose_move(self, board: str, extra):
        legal_moves = [
            (m.from_square // 8, m.from_square % 8, m.to_square // 8, m.to_square % 8)
            for m in board.legal_moves
        ]
        board_tensor = utils.convert_board_to_tensor(board)

        scores = self.model(board_tensor, extra, legal_moves)

        target_move = scores[max(scores)]
        from_row, from_col, to_row, to_col = target_move
        from_square = from_row * 8 + from_col
        to_square = to_row * 8 + to_col
        idx = from_square * 64 + to_square

        return board.legal_moves
