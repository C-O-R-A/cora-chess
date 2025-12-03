import chess
import numpy as np

def move_to_board_coord(move):
    from_coord = move.from_square // 8
    to_coord = move.to_square // 8
    return np.array([from_coord, to_coord])
