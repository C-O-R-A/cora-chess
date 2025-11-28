import sys
import os

sys.path.append("/home/matth/Desktop/Colossus/Software/chess/chess")

from data.compile_data import NeuralNet, ChessDataset, Load_player_data, convert_board_to_tensor
from visualize import visualize_model_weights_and_graph, plot_heatmap, plot_histogram

import chess
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

magnus_data = Load_player_data('Carlsen', 
                 'game_data/Carlsen, Magnus/Carlsen_moves.csv', 
                 'game_data/Carlsen, Magnus/Carlsen_game_info.csv', 
                 'game_data/Carlsen, Magnus/eco_codes.csv')
foldername = 'magnus_120'

# Predict move function
def predict_move(model, board_fen, extra_features, device):
    model.eval()
    board_tensor = convert_board_to_tensor(board_fen)
    # Assume the player is white for this example
    board_tensor = torch.tensor(board_tensor, dtype=torch.float).unsqueeze(0).to(device)  
    extra_tensor = torch.tensor(extra_features, dtype=torch.float).unsqueeze(0).to(device)

    # print(board_tensor)
    # print(extra_tensor)
    
    with torch.no_grad():
        from_logits, to_logits = model(board_tensor, extra_tensor)
        from_probs = torch.softmax(from_logits, dim=1)
        to_probs = torch.softmax(to_logits, dim=1)

    from_move = torch.argmax(from_probs).item()
    to_move = torch.argmax(to_probs).item()

    move = chess.Move(from_square=from_move, to_square=to_move)
    return move.uci()

# Predict a move and plot distribution
def sample_predict_and_plot(model, device, modelname):
    test_fen = "r1bqkNr1/ppp2pp1/2n4n/3pp3/8/3PP1P1/PPPN1PBP/R1BQK2R"
    test_extra_features = [18000, 3, 0]

    predicted_move = predict_move(model, test_fen, test_extra_features, device)
    print(f"Predicted move: {predicted_move}")

    def get_probs(model, board_fen, extra_features, device):
        model.eval()
        board_tensor = convert_board_to_tensor(board_fen)
        board_tensor = torch.tensor(board_tensor, dtype=torch.float).unsqueeze(0).to(device)
        extra_tensor = torch.tensor(extra_features, dtype=torch.float).unsqueeze(0).to(device)
        
        with torch.no_grad():
            from_logits, to_logits = model(board_tensor, extra_tensor)
            from_probs = torch.softmax(from_logits, dim=1).cpu().numpy().flatten()
            to_probs = torch.softmax(to_logits, dim=1).cpu().numpy().flatten()
        return from_probs, to_probs

    from_probs, to_probs = get_probs(model, test_fen, test_extra_features, device)
    plot_histogram(from_probs, 'from probs', f'figures/{modelname}/From_hist')
    plot_histogram(to_probs, 'to probs', f'figures/{modelname}/To_hist')
    plot_heatmap(from_probs, 'from probs', f'figures/{modelname}/From_heat')
    plot_heatmap(to_probs, 'to probs', f'figures/{modelname}/To_heat')

def train_model(model, num_epochs, modelname, dataset, plot=True):
    data_train = ChessDataset(dataset)
    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, 
                                drop_last=True, num_workers=4, pin_memory=True)
    try:
        os.makedirs(f'figures/{modelname}')
    except: 
        print(f'Directory {modelname} already exists')

    criterion = nn.CrossEntropyLoss()
    model = NeuralNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    num_epochs = 500
    losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in data_train_loader:
            inputs, targets = batch
            board_tensors = inputs['board'].to(device)
            extra_features = inputs['extra'].to(device)
            from_targets = targets[0].to(device)
            to_targets = targets[1].to(device)

            optimizer.zero_grad()
            from_logits, to_logits = model(board_tensors, extra_features)

            loss_from = criterion(from_logits, from_targets)
            loss_to = criterion(to_logits, to_targets)

            loss = loss_from + loss_to
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses.append(running_loss/len(data_train_loader))

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(data_train_loader)}")
        torch.save(model.state_dict(), f'models/{modelname}.pth')

    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title(f'Losses per batch')
    plt.xlabel('epoch')
    plt.ylabel('Losses')
    plt.savefig(f"figures/{modelname}/Losses.png")
    plt.show()

def visualize_trained_model(model, device, modelname):   
    model.load_state_dict(torch.load(f'models/{modelname}.pth', map_location=device))
    model.eval()
    visualize_model_weights_and_graph(f'models/{modelname}.pth', f'figures/{modelname}' )
    sample_predict_and_plot(model, device)