import tkinter as tk
# import levels
import json
import time
import torch
from torch import nn
import numpy as np
import pandas as pd


device = (
    # "cuda"
    # if torch.cuda.is_available()
    # else 
    "cpu"
)
# print(f"Using {device} device")

with open('config.json') as f:
    config = json.load(f)


NUM_INPUT=config['NUM_INPUT']
NUM_OUTPUT=config['NUM_OUTPUT']
LEN_TRAIN_DATA=config['LEN_TRAIN_DATA']

AAA=0



class ActorCritic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temp = 5
        self.dist=nn.Sequential(
            torch.nn.LayerNorm(NUM_INPUT//2),
            nn.Linear(NUM_INPUT//2, 25),
            nn.GELU(),
            nn.Linear(25, 25),
        )
        self.labels=nn.Sequential(
            nn.Linear(NUM_INPUT//2, 25),
            nn.Sigmoid(),
            nn.Linear(25, 25),
        )
        self.unite = nn.Sequential(
            # nn.GLU(),
            nn.Linear(25*2, 25),
            nn.GELU(),
        )

        self.out_actor = nn.Sequential(
            nn.Linear(25, 25),
            nn.GELU(),
            nn.Linear(25, 25),
            nn.GELU(),
            nn.Linear(25, NUM_OUTPUT),
        )
        self.out_critic = nn.Sequential(
            # nn.Linear(25, 25),
            # nn.GELU(),
            nn.Linear(25, 1),
        )
    def countCommonPart(self, x: torch.Tensor):
        out1 = self.dist(x[:NUM_INPUT//2])
        out2 = self.labels(x[NUM_INPUT//2:])
        return self.unite(torch.cat([out1, out2], dim=0))
    def forward_critic(self, x):
        o1 = self.countCommonPart(x)
        return self.out_critic(o1)
    def forward_actor(self, x):
        o1 = self.countCommonPart(x)
        return self.out_actor(o1)/self.temp
    def forward(self, x):
        o1 = self.countCommonPart(x)
        out_a = self.out_actor(o1)
        out_c = self.out_critic(o1)
        return out_a/self.temp, out_c


class Player:
    def __init__(self, actor_loss_fn, critic_loss_fn, lr, momentum) -> None:
        import mech
        self.scene: mech.Scene=None
        self.action = None
        self.isTraining=True
        self.countToTrain=0

        self.actor_critic=ActorCritic()
        self.actor_critic.to(device)
        self.optimizer1=torch.optim.NAdam(self.actor_critic.parameters(), lr=0.001)
        self.loss_fn1=actor_loss_fn
        self.action_prediction=None
        self.num_act=-1

        self.optimizer2=torch.optim.NAdam(self.actor_critic.parameters(), lr=0.005)
        self.loss_fn2=critic_loss_fn
        self.criticPastState = None
        self.criticPresentState = None

        self.envPastState = None
        self.envPresentState = None
    def get_IsTraining(self):
        return self.isTraining
    def change_IsTraining(self):
        self.isTraining=not self.isTraining
        if self.isTraining:
            self.actor_critic.train()
        else:
            self.actor_critic.eval()
    def get_countToTrain(self):
        return self.countToTrain
    def set_countToTrain(self, c=None):
        if c is None:
            self.countToTrain+=1
        else:
            self.countToTrain=c
    def fit(self, rew):
        if not self.get_IsTraining():
            return
        self.actor_critic.train()
        if (self.criticPresentState is None) or (self.criticPastState is None) or (self.envPresentState is None) or (self.envPastState is None):
            return
        if self.get_countToTrain()==0:
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()

        self.set_countToTrain()

        a_pi = torch.zeros(NUM_OUTPUT, device=device, dtype=torch.float32)
        a_pi[self.num_act] = rew+self.criticPresentState-self.criticPastState

        loss2: torch.Tensor = self.loss_fn2(self.criticPastState, a_pi[self.num_act:self.num_act+1])+self.loss_fn1(self.action_prediction, a_pi)
        loss2.backward()

        # a_pi[self.num_act] = rew+self.actor_critic.forward_critic(self.envPresentState)-self.actor_critic.forward_critic(self.envPastState)
        
        # loss1 = self.loss_fn1(self.action_prediction, a_pi)
        # loss1.backward()


        if self.get_countToTrain()==LEN_TRAIN_DATA:
            self.optimizer1.step()
            self.optimizer2.step()
            self.set_countToTrain(0)

    def predict_proba(self, X):
        return torch.nn.Softmax(-1)(self.actor_critic.forward_actor(X))

    def play(self, env):
        global AAA
        AAA+=1
        self.envPastState=self.envPresentState
        self.envPresentState=env

        if self.envPastState is not None:
            self.criticPastState = self.actor_critic.forward_critic(self.envPastState)
        else:
            self.criticPastState=None
        self.action_prediction, self.criticPresentState = self.actor_critic(env)
        prob = torch.nn.functional.softmax(self.action_prediction, -1)
        t = torch.rand(1)[0]
        self.num_act=-1
        while t>0:
            self.num_act+=1
            t-=prob[self.num_act]
        if AAA%50==0:
            print(f"distrib = {prob}; action = {self.num_act}; value = {self.criticPresentState[0].item():.2f}")
        # num_act=[np.random.choice([i for i in range(NUM_OUTPUT)], 
        #                          p=np.array(turn.iloc[0])/sum(turn.iloc[0]))]
        {0: lambda: self.scene.master.event_generate("<Key>", keysym="a"),
        1: lambda: self.scene.master.event_generate("<Key>", keysym="d"),
        2: lambda: self.scene.master.event_generate("<Key>", keysym="space"),
        3: self.scene.generate_UpLeft,
        4: self.scene.generate_UpRight,
        5: lambda: self.scene.master.event_generate("<KeyRelease>", keysym="a"),
        6: lambda: self.scene.master.event_generate("<KeyRelease>", keysym="d"),
        7: lambda: func_pass}[self.num_act]()

    def reset(self):
        self.action = None
        self.optimizer1.step()
        self.optimizer2.step()
        self.set_countToTrain(0)

        self.action_prediction=None
        self.num_act=-1

        self.criticPastState = None
        self.criticPresentState = None

        self.envPastState = None
        self.envPresentState = None

def func_pass():
    pass
