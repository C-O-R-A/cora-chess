import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchviz import make_dot
from neural_net import NeuralNet  # your model definition


def print_model_params(state_dict):
    print("Model parameters:")
    for name, param in state_dict.items():
        print(f"{name}: {param.shape}")

def find_flatten_size(model, dummy_board, dummy_extra):
    with torch.no_grad():
        x = model.input_conv(dummy_board)
        for block in model.res_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        print("Flattened conv feature vector shape:", x.shape)

        extra_features_out = model.fc_extra(dummy_extra)

        # Check extra features shape
        assert extra_features_out.shape[1] == model.fc_from.in_features - x.shape[1], \
            f"Expected extra features shape {(1, model.fc_from.in_features - x.shape[1])}, got {extra_features_out.shape}"

        return x.shape[1]

def visualize_model_weights_and_graph(
    model_path,
    modelfigures,
    layer_name='fc_from.weight',
    dummy_board_shape=(1, 6, 8, 8),
    dummy_extra_shape=(1, 3)  
):
    device = torch.device('cpu')

    model = NeuralNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print_model_params(model.state_dict())

    dummy_board = torch.randn(dummy_board_shape).to(device)
    dummy_extra = torch.randn(dummy_extra_shape).to(device)

    flatten_size = find_flatten_size(model, dummy_board, dummy_extra)
    expected_total_size = flatten_size + dummy_extra.shape[1]
    print(f"Expected FC input size: {expected_total_size}")

    output_from, output_to = model(dummy_board, dummy_extra)

    # Visualize weights of the specified layer
    if layer_name in model.state_dict():
        weights = model.state_dict()[layer_name].cpu().numpy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights, cmap='viridis')
        plt.title(f'Weight matrix of {layer_name}')
        plt.xlabel('Input Neurons')
        plt.ylabel('Output Neurons')
        plt.savefig(f'{modelfigures}/weight_matrix_heatmap.png')
        
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.hist(weights.flatten(), bins=100)
        plt.title(f'Weight distribution of {layer_name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.savefig(f'{modelfigures}/weight_distribution_histogram.png')
        plt.show()
    else:
        print(f"Layer '{layer_name}' not found in model state dict.")

    dot = make_dot((output_from, output_to), params=dict(model.named_parameters()))
    dot.render(f'{modelfigures}/model_graph', format="png")
    print("Saved model computation graph as 'model_graph.png'")

def plot_heatmap(probs, title, filename):
    probs_grid = probs.reshape(8,8)

    plt.figure(figsize=(7,6))
    ax = sns.heatmap(probs_grid, annot=False, cmap="viridis", cbar=True, square=True)
    plt.title(title)
    plt.xlabel("File (a-h)")
    plt.ylabel("Rank (1-8)")
    ax.invert_yaxis()  # So rank 1 is at bottom like a chessboard
    plt.savefig(filename)
    plt.show()

def plot_histogram(probs, title, filename):
    plt.figure(figsize=(7,5))
    plt.hist(probs, bins=20, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.show()