import numpy as np
import chess

def board_info(square_size, verbose=False) -> dict:
    square_dict = {}

    for i, square in enumerate(chess.SQUARE_NAMES):
        square = chess.square_name(chess.square_mirror(chess.Square(i)))
        row = i // 8
        col = i % 8

        x_l = col * square_size
        x_u = (col + 1) * square_size
        y_l = row * square_size
        y_u = (row + 1) * square_size

        _file = chess.square_file(i)
        _rank = chess.square_rank(i)

        square_dict[square] = {
            
            # Immutable characteristics
            "x_lower": x_l,
            "x_upper": x_u,
            "y_lower": y_l,
            "y_upper": y_u,
            "position": [0.5*(x_l +x_u), 0.5*(y_l +y_u)],
            "idx": 64 - i,
            "color": 'white' if (_file + _rank) % 2 == 1 else  'black',

            # Dynamic values
            "occupied": False,
            "noise_score": 0,
        }

    if verbose:
        print(square_dict)

    return square_dict