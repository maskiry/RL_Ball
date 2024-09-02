import tkinter as tk
# import levels
import json
import time
import torch
from torch import nn
import numpy as np
import pandas as pd
import player
import mech


class  GameFactory:
    def __init__(self, player: player.Player | None = None) -> None:
        self.window = mech.MainWindow()
        self.player=player
        if self.player:
            self.player.scene=self.window.scene

        self.do_motion()
    def do_motion(self):
        if not self.window.scene.pause:
            self.act_time = time.time()
            if (self.player):
                self.player.actor_critic.temp=self.window.model_p['temp']/10
                self.player.isTraining=self.window.model_p['training mode']
                self.player.play(self.window.scene.get_env())
                rev = self.window.scene.update_scene()
                if (self.player.isTraining):
                    self.player.fit(rev)
                if self.window.scene.restartFlag:
                    self.window.scene.restartFlag=False
                    self.player.reset()
            else:
                self.window.scene.update_scene()
        self.window.after(int(19-(time.time()-self.act_time)*1000), self.do_motion)
    def start(self):
        self.window.mainloop()