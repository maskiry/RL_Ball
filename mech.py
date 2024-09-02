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

with open('config.json') as f:
    config = json.load(f)

SPEED=config['SPEED']
G = config['G']
MAX_SPEED=config['MAX_SPEED']
REWARD_VICT=config['REWARD_VICT']
PENALTY_LOSE=config['PENALTY_LOSE']
NUM_INPUT=config['NUM_INPUT'] #число входных нейронов, четное
LVL_TIME_LIMIT = config['LVL_TIME_LIMIT'] #в сек




class MainWindow(tk.Tk):
    def __init__(self) -> None:
        print("------Creating main window------")
        super().__init__()
        self.bind("<Escape>", lambda x: self.destroy())
        self.state('zoomed')
        self.overrideredirect(1)
        self.scene=Scene(self)
        self.model_p = {"temp": 50.,
                        "training mode": True}

        self.menu: tk.Menu=tk.Menu(self)
        self.menu.add_cascade(label="Pause", command=self.pause_click)
        self.menu.add_cascade(label="Lvl 1: 0:00.0/0:30.0")
        self.menu.add_separator()
        self.parameters_menu = tk.Menu(self, tearoff=0)
        self.temp_menu = tk.Menu(self, tearoff=0)
        self.temp_menu.add_cascade(label="+", command=lambda: self.change_temp(10))
        self.temp_menu.add_cascade(label="-", command=lambda: self.change_temp(-10))
        self.parameters_menu.add_cascade(label=f"Temp: {self.model_p['temp']}", menu=self.temp_menu)
        self.parameters_menu.add_cascade(label=f"Training:{self.model_p['training mode']}", command=lambda: self.change_isTraining())
        self.menu.add_cascade(label="Model parameters", menu=self.parameters_menu)
        self.config(menu=self.menu)
        self.timeInPause=0
        print("------Created main window------")
    def get_scene(self):
        return self.scene
    def pause_click(self):
        self.menu.entryconfigure(1, label="Pause" if self.scene.pause else "Continue")
        self.scene.set_pause(not self.scene.pause)
    def change_temp(self, delta_t):
        self.model_p['temp']=max([1, (self.model_p['temp']+delta_t)])
        self.parameters_menu.entryconfigure(1, label=f"Temp: {self.model_p['temp']}")
    def change_isTraining(self):
        self.model_p['training mode'] = not self.model_p['training mode']
        self.parameters_menu.entryconfigure(2, label=f"Training:{self.model_p['training mode']}")
    # def get_menu(self):
    #     return self.menu

class Object:
    def __init__(self, scene, movable,  type: str, fill: str, tag: str, coords: list, velocity=[0,0]) -> None:
        """coords: [x0,y0,x1,y1] if oval, line or rect"""
        self.scene: Scene=scene
        self.movable=movable
        self.coords=coords
        self.velocity=velocity
        self.tag=tag
        self.obj={type=="oval": lambda: self.scene.create_oval(coords, fill=fill, tags=tag),
                  type=="line": lambda: self.scene.create_line(coords, fill=fill, tags=tag, width=15),
                  type=="rect": lambda: self.scene.create_rectangle(coords, fill=fill, tags=tag)}[True]()
        self.usageTime = time.time()
    def get_move(self):
        return self.movable
    def get_coord(self):
        return self.coords
    def get_obj(self):
        return self.obj
    def get_vel(self):
        return self.velocity
    def change_vel(self, delta_vel: list[float]):
        if (not self.movable):
            return
        if abs(self.velocity[0]+delta_vel[0])<MAX_SPEED:
            self.velocity[0]+=delta_vel[0]
        if abs(self.velocity[1]+delta_vel[1])<MAX_SPEED*2:
            if delta_vel[1]<0:
                all_tags=self.scene.find_all()
                hero=self.scene.find_withtag(self.tag)
                hor=set(self.scene.find_withtag("horizontal"))
                below=set([tag for tag in all_tags if self.scene.coords(tag)[1]>self.scene.coords(hero)[3]])
                x0,y0,x1,y1=self.scene.bbox(hero)
                overlap=set(self.scene.find_overlapping(x0,y0,x1,y1))
                if len(overlap & below & hor)>0:
                    self.velocity[1]+=delta_vel[1]
            else:
                if (self.velocity[1]+delta_vel[1])<MAX_SPEED*1.5:
                    self.velocity[1]+=delta_vel[1]
        # elif abs(self.velocity[1]+delta_vel[1])<MAX_SPEED:
        #     self.velocity[1]+=delta_vel[1]


class Scene(tk.Canvas):
    def __init__(self, master: MainWindow, bg="sky blue", pause=False) -> None:
        print("------Creating scene------")
        super().__init__(master, background=bg)
        self.pack(fill=tk.BOTH, expand=1)
        self.objects: dict[str, Object]=dict()
        self.pause=pause
        self.pauseTime = 0
        self.pauseStartTime=0
        self.restartFlag=True
        self.lvl=Level(master, self)
        self.distToEx=((self.coords(self.objects["hero0"].get_obj())[0]-
                    self.coords(self.objects["exit0"].get_obj())[0])**2+
                    (self.coords(self.objects["hero0"].get_obj())[1]-
                    self.coords(self.objects["exit0"].get_obj())[1])**2)**0.5
        self.master.bind("<KeyPress>", lambda ev: self.KeyPressed(ev))
        self.master.bind("<KeyRelease>", lambda ev: self.KeyReleased(ev))
        self.jump_action = False
        print("------Created scene------")
    def get_lvl(self):
        return self.lvl
    def get_obj(self):
        return self.objects
    def find_dist(self, quart, X, Y, alpha):
        objects: list[int]=self.find_all()
        if (alpha-90)%180==0:
            #Если прямая вертикальна, то k=inf
            k=None
        elif (alpha)%180==0:
            #Если прямая горизонтальна, то k=0
            k=0
            b=Y
        else:
            k=np.tan(alpha*2*np.pi/360)
            b=Y-k*X
        dist=[]
        for i in range(len(objects)):
            #Пробегаем по всем объектам на экране
            obj_tag=self.gettags(objects[i])[0]
            if obj_tag=="mainPerson":
                #Если гг, то идем дальше
                continue
            # obj_type={obj_tag=="exit": 1,
            #           obj_tag=="horizontal": 2,
            #           obj_tag=="vertical": 3,
            #           obj_tag=="obstacle": 4}[True]
            obj_type=(obj_tag=="obstacle")+0. #1, если опасно, 0, если неопасно

            coords=self.coords(objects[i])
            if k is None:
                if X>=min(coords[::2]) and X<=max(coords[::2]):
                    #Если гг над/под объектом, то расстояние зависит от у
                    #Это работает, потому что obstacle симметричный
                    if coords[2]!=coords[0]:
                        y=(coords[1]*(X-coords[0])/(coords[2]-coords[0])+
                        coords[3]*(coords[2]-X)/(coords[2]-coords[0]))
                    else:
                        y=coords[1]
                    x=X
                else:
                    continue
                if alpha==90 and y-Y<=0:
                    continue
                if alpha==270 and y-Y>=0:
                    continue
            elif k==0:
                if Y>=min(coords[1::2]) and Y<=max(coords[1::2]):
                    y=Y
                    x=(coords[0]+coords[-2])/2
                else:
                    continue
                if alpha==0 and x-X<=0:
                    continue
                if alpha==180 and x-X>=0:
                    continue
            else:
                if obj_tag=="obstacle":
                    intersec1 = self.find_intersec(k, b, coords[:4])
                    intersec2 = self.find_intersec(k, b, coords[2:])
                    if intersec1 is None:
                        if intersec2 is None:
                            continue
                        x,y=intersec2
                    elif intersec2 is None:
                        x,y=intersec1
                    else:
                        x,y=(intersec1[0]+intersec2[0])/2,(intersec1[1]+intersec2[1])/2,
                else:
                    intersec = self.find_intersec(k, b, coords)
                    if intersec is None:
                        continue
                    x,y=intersec
                if (y-Y)-k*(x-X)>0.001:
                    continue
            # if quart==self.get_quart(x-X, y-Y):
            # print(f"delta_coords = {x-X}, {y-Y}; alpha={alpha}, quart={quart};{self.get_quart(x-X, y-Y)}")
            dist.append([(np.sqrt((x-X)**2+(y-Y)**2)), obj_type])
        if len(dist)>0:
            # print(dist)
            return min(dist, key=lambda j: j[0])
        else:
            # print("КОСЯК")
            return [1500., 0]
    def get_quart(self, x, y):
        """Delete this method?"""
        if x>0 and y>=0:
            return 1
        elif x<=0 and y>0:
            return 2
        elif x<0 and y<=0:
            return 3
        else:
            return 4
    def find_intersec(self, k, b, coords):
        if coords[0]==coords[2]:
            x0=coords[0]
            y0=k*x0+b
            if y0>=min(coords[1::2]) and y0<=max(coords[1::2]):
                return [x0, y0]
            else:
                return None
        k1=(coords[1]-coords[3])/(coords[0]-coords[2])
        b1=coords[1]-k1*coords[0]
        if abs(k1-k)<0.001:
            if abs(b1-b)<0.001:
                return [(coords[0]+coords[2])/2, (coords[1]+coords[3])/2]
            return None
        x0=-(b1-b)/(k1-k)
        y0=k*x0+b
        if x0<min(coords[::2]) or x0>max(coords[::2]):
            return None
        if y0<min(coords[1::2]) or y0>max(coords[1::2]):
            return None
        return [x0,y0]
    def get_env(self)->torch.Tensor:
        hero=self.find_withtag("mainPerson")
        x0, y0, x1, y1=self.coords(hero)
        X, Y=(x0+x1)/2, (y0+y1)/2
        inputs = torch.zeros(NUM_INPUT, dtype = torch.float32, device = device)
        for i in range(0, NUM_INPUT//2, 1):
            alpha=720/NUM_INPUT*i
            # if (alpha-90)%180==0:
            #     continue
            if alpha<90:
                quart=1
            elif alpha<180:
                quart=2
            elif alpha<270:
                quart=3
            else:
                quart=4
            d, num_obj=self.find_dist(quart, X, Y, alpha)
            inputs[i]=d
            inputs[i+NUM_INPUT//2]=num_obj
        return inputs
    def set_pause(self, pause):
        self.pause=pause
        if not self.pause:
            for key in self.objects.keys():
                if self.objects[key].get_move():
                    self.objects[key].usageTime=time.time()
            self.pauseTime+=time.time()-self.pauseStartTime
        else:
            self.pauseStartTime = time.time()
    def add_obj(self, key, obj: Object):
        self.objects[key]=obj
    def clear_obj(self):
        self.objects: dict[str, Object]=dict()
    def checkCollision(self):
        all_tags=self.find_all()
        hero=self.find_withtag("mainPerson")
        hor=set(self.find_withtag("horizontal"))
        ver=set(self.find_withtag("vertical"))
        ex=set(self.find_withtag("exit"))
        obs=set(self.find_withtag("obstacle"))
        above=set([tag for tag in all_tags if self.coords(tag)[3]<self.coords(hero)[1]])
        below=set([tag for tag in all_tags if self.coords(tag)[1]>self.coords(hero)[3]])
        left=set([tag for tag in all_tags if self.coords(tag)[2]<self.coords(hero)[0]])
        right=set([tag for tag in all_tags if self.coords(tag)[0]>self.coords(hero)[2]])
        x0,y0,x1,y1=self.bbox(hero)
        overlap=set(self.find_overlapping(x0,y0,x1,y1))
        if len(overlap & below & hor)>0:
            self.objects["hero0"].change_vel([0, min([0, -self.objects["hero0"].get_vel()[1]])])
        # else:
        #     print(G*(time.time()-self.objects["hero0"].usageTime))
        #     self.objects["hero0"].change_vel([0, G*(time.time()-self.objects["hero0"].usageTime)])
        if len(overlap & above & hor)>0:
            self.objects["hero0"].change_vel([0, max([0, -self.objects["hero0"].get_vel()[1]])])
        if len(overlap & left & ver)>0:
            self.objects["hero0"].change_vel([max([0, -self.objects["hero0"].get_vel()[0]]), 0])
        if len(overlap & right & ver)>0:
            self.objects["hero0"].change_vel([min([0, -self.objects["hero0"].get_vel()[0]]), 0])
        if len(overlap & obs)>0:
            """LOOOSE"""
            self.set_pause(True)
            self.lvl.change_lvl(self.lvl.get_numLev())
            return PENALTY_LOSE
        if len(overlap & ex)>0:
            """VICTORY!!!"""
            self.set_pause(True)
            if self.lvl.get_numLev()+1==self.lvl.maxNum:
                self.master.change_isTraining()
            self.lvl.change_lvl(self.lvl.get_numLev()+1)
            return REWARD_VICT
        return 0
    def KeyPressed(self, e: tk.Event):
        key=e.keysym
        {key=="space": lambda: self.objects["hero0"].change_vel([0, -SPEED*2]),
         key=="a": lambda: self.objects["hero0"].change_vel([-SPEED, 0]),
         key=="d": lambda: self.objects["hero0"].change_vel([SPEED, 0]),
         key!="a" and key!="d" and key!="space": func_pass}[True]()
        self.jump_action = (key=="space")
        self.checkCollision()
    def KeyReleased(self, e: tk.Event):
        key=e.keysym
        {key=="a": lambda: self.objects["hero0"].change_vel([-self.objects["hero0"].get_vel()[0], 0]),
         key=="d": lambda: self.objects["hero0"].change_vel([-self.objects["hero0"].get_vel()[0], 0]),
         key!="a" and key!="d": func_pass}[True]()
        self.checkCollision()
    def update_scene(self):
        self.master.menu.entryconfigure(2, label=f"Lvl {self.lvl.get_numLev()+1}: 0:{(time.time()-self.lvl.creationTime-self.pauseTime):.1f}/0:30.0")
        if time.time()-self.lvl.creationTime-self.pauseTime>LVL_TIME_LIMIT:
            """LOOOSE"""
            self.set_pause(True)
            self.lvl.change_lvl(self.lvl.get_numLev())
            self.restartFlag=True
            return -1

        all_tags=self.find_all()
        hero=self.find_withtag("mainPerson")
        hor=set(self.find_withtag("horizontal"))
        below=set([tag for tag in all_tags if self.coords(tag)[1]>self.coords(hero)[3]])
        x0,y0,x1,y1=self.bbox(hero)
        overlap=set(self.find_overlapping(x0,y0,x1,y1))
        if len(overlap & below & hor)>0:
            self.objects["hero0"].change_vel([0, min([0, -self.objects["hero0"].get_vel()[1]])])
        else:
            self.objects["hero0"].change_vel([0, G*(time.time()-self.objects["hero0"].usageTime)])
        for key in self.objects.keys():
            if self.objects[key].get_move():
                pass_time = min([0.022, time.time()-self.objects[key].usageTime])
                self.move(self.objects[key].get_obj(), 
                          self.objects[key].get_vel()[0]*(pass_time), 
                          self.objects[key].get_vel()[1]*(pass_time))
                self.objects[key].usageTime=time.time()
        col=self.checkCollision()
        if col==REWARD_VICT or col==PENALTY_LOSE:
            self.restartFlag=True
            return col

        dist=((self.coords(self.objects["hero0"].get_obj())[0]-
                self.coords(self.objects["exit0"].get_obj())[0])**2+
                (self.coords(self.objects["hero0"].get_obj())[1]-
                self.coords(self.objects["exit0"].get_obj())[1])**2)**0.5
        delta_d=(self.distToEx-dist-(
            abs(self.distToEx-dist)<0.5)+0.)*(1-self.jump_action*0.1)/2#/self.master.winfo_screenwidth()#>0, если приближается к выходу, иначе <0
        self.jump_action=False
        # print(self.distToEx-dist)
        self.distToEx = dist
        return delta_d
    def generate_UpLeft(self):
        self.master.event_generate("<Key>", keysym="a")
        self.master.event_generate("<Key>", keysym="space")
    def generate_UpRight(self):
        self.master.event_generate("<Key>", keysym="d")
        self.master.event_generate("<Key>", keysym="space")



class Level:
    def __init__(self, master: MainWindow, scene: Scene, num=0) -> None:
        self.master=master
        self.scene=scene
        self.num=num
        with open('levels.json') as f:
            templates = json.load(f)
        self.maxNum=len(templates.keys())
        self.creationTime = None
        self.create_lvl()
    def change_lvl(self, num):
        self.scene.delete("all")
        self.num=num%self.maxNum
        self.create_lvl()
        self.scene.set_pause(False)
        print("Let's go")
        self.scene.pauseTime=0
        self.scene.pauseStartTime=0
        # self.scene.do_motion()
    def get_numLev(self):
        return self.num
    def get_maxLev(self):
        return self.maxNum
    def create_lvl(self):
        with open('levels.json') as f:
            templates = json.load(f)
        # self.maxNum=len(templates.keys())
        for key in templates[f"{self.num}"].keys():
            params=templates[f"{self.num}"][key]
            i=0
            for param in params:
                self.scene.add_obj(f"{key}{i}", Object(self.scene, param["movable"], param["type"], param["color"], param["tags"], param["coord"]))
                i+=1
        self.creationTime = time.time()
        # self.scene.do_motion()





def func_pass():
    pass

