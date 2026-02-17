import chess
import helpers as h
import numpy as np

class board:
    def __init__(self, marker_size):
        self.Board = chess.Board()
        self.position = np.zeros(3)  
        self.orientation = np.zeros(3)
        self.marker_size = marker_size
        self.markers_3d = None
        self.markers_2d = None
        self.square_size = 0.02  # 2cm default
        self.squares = None
        return

    def update_board_state(
        self, move:str
    ):  # should be called once every turn
        self.Board.push(chess.Move.from_uci(move))
        return
    
    def update_positions(
        self, markers: dict
    ):  # should be called continuously to keep track of board position
        self.markers_3d = markers
        self.position = (markers["marker_0"]["pose"] + markers["marker_1"]["pose"]) / 2
        self.rotation = None
        return
    
    def set_board_info(self, square_size):
        self.squares = h.board_info(square_size=square_size)

    def get_board_state(self):
        return self.Board.fen()