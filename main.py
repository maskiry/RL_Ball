import tkinter as tk
import torch
import player
import Game

if __name__=="__main__":
    pl = player.Player(torch.nn.CrossEntropyLoss(), torch.nn.MSELoss(), 0.01, 0.0005)
    game = Game.GameFactory(pl)
    game.start()