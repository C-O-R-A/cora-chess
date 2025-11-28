import chess
import torch 
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
from compile_data import process_color, process_date_played, process_move_number, convert_board_to_tensor, convert_move_to_tensor, expand_tensor


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

    # Filter board positions after opponent moves
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
    combined_data.to_csv("magnus_move_data_6.csv", index=False)

    # # For debugging, print fen types
    # print(play_data['fen'].apply(type).value_counts())

    return combined_data, eco_mapping

class ChessDataset(Dataset):
    def __init__(self, data, use_output_tensors=False):
        super(ChessDataset, self).__init__()
        self.data = data[0]
        self.eco_to_idx = data[1]
        self.use_output_tensors = use_output_tensors

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Board tensor
        board_tensor = convert_board_to_tensor(row['fen'])

        # Extra features
        extra_data = [
            # process_eco_code(row['eco'], self.eco_to_idx),
            process_date_played(row['date_played']),
            process_move_number(row['move_no_pair_move']),
            process_color(row['color_move']),
            # process_count(row['white_count']),
            # process_count(row['black_count'])
        ]

        input_data = expand_tensor(board_tensor, extra_data)

        # Move tensors
        from_layer, to_layer = convert_move_to_tensor(row['move'])
        move_obj = chess.Move.from_uci(row['move'])
        from_idx = move_obj.from_square  # 0-63
        to_idx = move_obj.to_square      # 0-63

        if self.use_output_tensors:
            move_target = {
                'from': torch.tensor(from_layer, dtype=torch.float),
                'to': torch.tensor(to_layer, dtype=torch.float)
            }

        else:
            move_target = (from_idx, to_idx)

        return input_data, move_target
        
class module(nn.Module):

    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)        
        x = self.bn1(x)
        x = self.activation1(x)        
        x = self.conv2(x)        
        x = self.bn2(x)
        x = x + x_input
        x = self.activation2(x)
        return x        

class NeuralNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200, extra_feature_dim=3):
        super().__init__()
        self.input_conv = nn.Conv2d(6, hidden_size, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(hidden_size)
        self.activation = nn.ReLU()

        self.res_blocks = nn.Sequential(
            *[module(hidden_size) for _ in range(hidden_layers)]
        )

        self.flatten = nn.Flatten()
        self.fc_input_size = hidden_size * 8 * 8

        self.fc_from = nn.Linear(self.fc_input_size + 64, 64)
        self.fc_to = nn.Linear(self.fc_input_size + 64, 64)
        self.fc_extra = nn.Linear(extra_feature_dim, 64)

    def forward(self, board_tensor, extra_features):
        x = self.input_conv(board_tensor)
        x = self.bn_input(x)
        x = self.activation(x)

        x = self.res_blocks(x)
        x = self.flatten(x)

        extra_features = self.fc_extra(extra_features)
        extra_features = self.activation(extra_features)

        # concatenate extra features
        x = torch.cat([x, extra_features], dim=1)

        from_logits = self.fc_from(x)
        to_logits = self.fc_to(x)

        return from_logits, to_logits