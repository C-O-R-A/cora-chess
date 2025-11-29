import chess
import torch 
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
from datetime import datetime

def Load_player_data(player_name, move_csv_path, game_csv_path, eco_csv_path):
    '''
    move data should contain move, game_id, move_no_pair, player, color
    board data should contain the rest
    '''

    # Load move and board data from the same CSV (assuming this is correct)
    move_data = pd.read_csv(move_csv_path, usecols=['game_id', 'move_no_pair', 'color', 'move', 'player'])
    board_data = pd.read_csv(move_csv_path, usecols=['game_id', 'move_no_pair', 'player', 'color', 'fen', 'white_count', 'black_count'])

    # Filter moves made by the target player
    move_data = move_data[move_data['player'].str.contains(player_name, na=False)]

    # Filter board positions before target player moves
    board_data = board_data[~board_data['player'].str.contains(player_name, na=False)]

    # Calculate corresponding board move_no_pair for each move:
    move_data_adj = move_data.copy()
    move_data_adj['board_move_no_pair'] = move_data_adj.apply(
        lambda row: row['move_no_pair'] - 1 if row['color'].lower() == 'white' else row['move_no_pair'],
        axis=1
    )

    # Identify first white moves (no prior board)
    first_white_moves = move_data_adj[move_data_adj['board_move_no_pair'] == 0]

    # Default starting board state
    starting_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
    starting_white_count = 16
    starting_black_count = 16

    if not first_white_moves.empty:
        starting_board_data = first_white_moves[['game_id']].copy()
        starting_board_data['move_no_pair'] = 0  # for matching
        starting_board_data['fen'] = starting_fen
        starting_board_data['white_count'] = starting_white_count
        starting_board_data['black_count'] = starting_black_count
        starting_board_data['player'] = 'default'
        starting_board_data['color'] = 'Black'  # set color so this matches board_data columns

        # Append default starting boards to board_data
        board_data = pd.concat([board_data, starting_board_data], ignore_index=True)

    # Make sure no missing columns before merge and drop player from board_data
    board_data = board_data.drop(columns=['player'])

    # Merge moves with boards on game_id and move number
    play_data = pd.merge(
        move_data_adj,
        board_data,
        left_on=['game_id', 'board_move_no_pair'],
        right_on=['game_id', 'move_no_pair'],
        how='left',
        suffixes=('_move', '_board')
    )

    # Load game metadata with game_id column for merging
    game_data = pd.read_csv(game_csv_path)
    # Assumption: game_data contains 'game_id', 'date_played', 'eco'
    if 'game_id' not in game_data.columns:
        raise ValueError("game_csv_path file must contain 'game_id' column")

    # Load ECO codes CSV and create mapping
    eco_df = pd.read_csv(eco_csv_path)
    eco_df = eco_df.sort_values(by='eco').reset_index(drop=True)
    eco_mapping = {eco: idx+1 for idx, eco in enumerate(eco_df['eco'].values)}

    # Merge game info to play_data on game_id
    combined_data = pd.merge(game_data, play_data, on='game_id', how='left')

    # Fill any missing values for important columns
    combined_data['fen'] = combined_data['fen'].fillna(starting_fen)
    combined_data['white_count'] = combined_data['white_count'].fillna(starting_white_count).astype(int)
    combined_data['black_count'] = combined_data['black_count'].fillna(starting_black_count).astype(int)
    combined_data['color_move'] = combined_data['color_move'].fillna('Unknown')

    combined_data['move'] = combined_data['move'].fillna('')  # or whatever default

    # Save combined data
    combined_data.to_csv("loaded_magnus_move_data.csv", index=False)

    # For debugging, print fen types
    print(play_data['fen'].apply(type).value_counts())

    return combined_data, eco_mapping

class ChessDataset(Dataset):
    def __init__(self, data):
        """
        data: tuple (DataFrame, eco_to_idx) or just DataFrame
        """
        self.data = data[0] if isinstance(data, tuple) else data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row['fen']
        board = chess.Board(fen)

        # Set correct turn
        board.turn = chess.WHITE if row['color_move'] == 'White' else chess.BLACK

        # Legal moves
        legal_moves = [(m.from_square, m.to_square) for m in board.legal_moves]
        legal_moves_tensor = torch.tensor(legal_moves, dtype=torch.float32)  # [N,2]

        # Target index in legal moves
        target_move = (chess.Move.from_uci(row['move']).from_square,
                       chess.Move.from_uci(row['move']).to_square)
        target_index = legal_moves.index(target_move)
        target_index = torch.tensor(target_index, dtype=torch.long)

        # Board tensor
        board_tensor = self.convert_board_to_tensor(fen)  # [6,8,8]

        # Extra features
        extra_features = torch.tensor([
            self.process_date_played(row['date_played']),
            int(row['move_no_pair_move']),
            0 if row['color_move'] == 'White' else 1
        ], dtype=torch.float32)

        return {
            "board": board_tensor,
            "extra": extra_features,
            "legal_moves": legal_moves_tensor,
            "target_index": target_index
        }
    
    def convert_board_to_tensor(self,board_fen):
        pieces = ['p', 'r', 'n', 'b', 'q', 'k']
        board = chess.Board(board_fen)
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

        return torch.tensor(np.stack(layers), dtype=torch.float32)
    
    def process_date_played(self, date_played):
        date_played = date_played.replace("??", "01")
        dt = datetime.strptime(date_played, "%Y.%m.%d")
        epoch = datetime(1970,1,1)
        delta_days = (dt - epoch).days
        return delta_days
        
class module(nn.Module):

    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.ReLU()
        
        self.layers = nn.Sequential(self.conv1, self.bn1, self.activation1, self.conv2, self.bn2)

        self.activation2 = nn.ReLU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.layers(x)
        x = x + x_input
        x = self.activation2(x)
        return x        

class NeuralNet(nn.Module):
    '''
    A convolutional neural network for predicting chess moves from a board tensor
    and auxiliary (non-spatial) features.

    The network processes a 6×8×8 input tensor representing piece placements across
    six channels (e.g., piece-type × color planes). It applies an initial convolution,
    a stack of residual blocks, then flattens the result and combines it with
    additional non-board features. 
    '''

    def __init__(self, hidden_layers=4, hidden_size=200, extra_feature_dim=3):
        super().__init__()
        '''
        Initialize the neural network.

        Args:
            hidden_layers (int):
                Number of residual blocks applied after the initial convolution.
            hidden_size (int):
                Number of feature channels in the convolutional and residual layers.
            extra_feature_dim (int):
                Dimension of the auxiliary non-board input feature vector
                (e.g., side-to-move, castling rights, move counters).

        Components created:
            • input_conv: first 3×3 convolution mapping 6 channels → hidden_size  
            • bn_input: batch normalization for the input convolution  
            • activation: shared ReLU activation  
            • res_blocks: a Sequential container of `hidden_layers` residual blocks  
            • flatten: flattens convolutional output to a vector  
            • fc_extra: linear layer that embeds auxiliary features to 64 units  

        '''
        self.input_conv = nn.Conv2d(6, hidden_size, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(hidden_size)
        self.activation = nn.ReLU()

        self.res_blocks = nn.Sequential(
            *[module(hidden_size) for _ in range(hidden_layers)]
        )

        self.flatten = nn.Flatten()
        self.fc_extra = nn.Linear(extra_feature_dim, 64)

    def forward(self, board_tensor, extra_features, legal_moves):
        '''
        Run a forward pass of the network.

        Args:
            board_tensor (Tensor):
                A float tensor of shape (batch_size, 6, 8, 8) representing the chess
                board. Each of the 6 channels typically encodes a piece type and color
                (e.g., white pawns, white pieces, black pawns, …).

            extra_features (Tensor):
                A tensor of shape (batch_size, extra_feature_dim) containing
                side-information not encoded spatially (e.g., castling rights,
                fifty-move counter, who's to move).
                
            legal_moves (Tensor): 
                A tensor of legal moves with square index coordinates
        Returns:

        '''
        x = self.input_conv(board_tensor)
        x = self.bn_input(x)
        x = self.activation(x)
        x = self.res_blocks(x)
        x = self.flatten(x)

        x_extra = self.fc_extra(extra_features)
        x_extra = self.activation(x_extra)

        # concatenate extra features
        x = torch.cat([x, x_extra], dim=1)

        return 

    
class ChessMoveSelector(nn.Module):
    def __init__(self, num_extra_features=3, board_embed_dim=256, move_embed_dim=128):
        super().__init__()
        # CNN for board
        self.cnn = nn.Sequential(
            nn.Conv2d(6,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, board_embed_dim), nn.ReLU()
        )
        self.extra_fc = nn.Linear(num_extra_features, board_embed_dim)
        self.move_fc = nn.Linear(2, move_embed_dim)
        self.combine_fc = nn.Linear(board_embed_dim + move_embed_dim, 1)

    def forward(self, board, extra, legal_moves_list):
        """
        board: [B,6,8,8]
        extra: [B,num_extra]
        legal_moves_list: list of length B, each [N_i,2]
        """
        B = board.size(0)
        scores_list = []

        board_emb = self.cnn(board)
        extra_emb = self.extra_fc(extra)
        board_emb = board_emb + extra_emb  # [B, board_embed_dim]

        for i in range(B):
            moves = legal_moves_list[i]           # [N_i,2]
            move_emb = self.move_fc(moves.float())  # [N_i, move_embed_dim]
            b_emb = board_emb[i].unsqueeze(0).expand(move_emb.size(0), -1)
            combined = torch.cat([b_emb, move_emb], dim=1)
            score = self.combine_fc(combined).squeeze(1)  # [N_i]
            probs = F.softmax(score, dim=0)               # softmax over legal moves
            scores_list.append(probs)

        return scores_list