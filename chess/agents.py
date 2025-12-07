import numpy
import utils
import torch.nn as nn

class Agent():
    def __init__(self,name, model, ):
        self.model = model
        return
    
    def choose_move(self, ):
        self.model(board, extra, legal_moves_list)