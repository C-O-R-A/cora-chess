import chess
import numpy as np

def move_to_sq(move):
    move = chess.Move.from_uci(move)
    to_sq_x = int(move.to_square // 8)
    to_sq_y = int(move.to_square % 8) 
    from_sq_x = int(move.from_square // 8) 
    from_sq_y = int(move.to_square % 8) 

    return (from_sq_x +1, from_sq_y +1, to_sq_x +1, to_sq_y +1)

# print(move_to_sq('e2e4'))