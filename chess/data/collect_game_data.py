import chess
import chess.pgn 
from pathlib import Path
import time
import csv

'''
Functions to collect game data manually from pgn files and to export said data to a .csv file
'''

folder_Magnus = Path("game_data/Carlsen, Magnus")
magnus_csv_path = "recorded_data/Carlsen_Magnus.csv"
folder_Nakamura = Path("game_data/Nakamura, Hikaru")

def gather_data(player_names, folder, csv_path):
    data = []
    for file in folder.iterdir():
        if file.is_file():
            if file.suffix == ".pgn":
                print("reading :", file)
                with file.open(encoding="utf-8") as pgn:
                    game_number = 0
                    while True:
                        game = chess.pgn.read_game(pgn)
                        if game is None:
                            break  # No more games in this file                    
                        
                        # print ("game: ", game)
                        white = game.headers["White"]
                        black = game.headers["Black"]
                        opponent_color = None
                        player_color = None
                        player_name = None      
                        opponent_name = None              

                        if white in player_names:
                            player_color = "white"
                            opponent_color = "black"
                            player_name = white
                            opponent_name = black

                        elif black in player_names:
                            player_color = "black"
                            opponent_color = "white"
                            player_name = black
                            opponent_name = white

                        else: 
                            print("Player not found in game headers")
                            print("White: ", white)
                            print("Black: ", black)
                            ValueError("Player not found in game headers")

                        moves = list(game.mainline_moves())
                        board = chess.Board()
                    
                        inputs = [] # opponent moves
                        outputs = [] # player moves
                        
                        # Store the moves in a way that is easy to use for training              
                        for i, move in enumerate(moves):

                            if (player_color == "white" and (i%2) == 0) or (player_color == "black" and i%2 != 0): 
                                # if its the players turn, record the move made
                                fen = board.fen()  
                                uci_move = move.uci()                              
                                outputs.append(move.uci())
                                data.append([fen, uci_move])

                            try:
                                board.push(move)
                            except Exception as e:
                                print(f"Illegal move {move.uci()} at move {i+1} in {file.name}: {e}")
                                print("Skipping this game.")
                                break

                        print ("Player_color: ", player_color, "       ", "Opponent_color: ", opponent_color)
                        print ("Player: ", player_name, "       ", "Opponent: ", opponent_name)
                        print ("Number of moves: ", max(len(outputs), len(inputs)))
                        print ("")
                        print (game_number)
                        game_number += 1
                        time.sleep(0.1)
    
    print("Finished reading all games")
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(["fen", "move"])
        writer.writerows(data)
    print("Finished writing data to CSV file")

gather_data(["Carlsen, Magnus", "Carlsen, Magnus", "Carlsen, M.", "Carlsen, M.", "Carlsen Magnus (NOR)", "Carlsen,M"], 
            folder_Magnus, 
            magnus_csv_path)
#gather_data("Nakamura, Hikaru", folder_Nakamura)
