import sys
BIN = '../'
sys.path.append(BIN)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from fastai import data_block, basic_train, basic_data
from fastai.callbacks import ActivationStats
import fastai
import matplotlib as mpl

class AE(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.en1= nn.Linear(4,200,bias=True)
    self.en2= nn.Linear(200,200,bias=True)
    self.en3= nn.Linear(200,100,bias=True)
    self.en4= nn.Linear(100,100,bias=True)
    self.en5= nn.Linear(100,50,bias=True)
    self.en6 = nn.Linear(50,25,bias=True)
    self.en7 = nn.Linear(25,3,bias=True)
    self.de1 = nn.Linear(3,25,bias=True)
    self.de2 = nn.Linear(25,50,bias=True)
    self.de3 = nn.Linear(50,100,bias=True)
    self.de4 = nn.Linear(100,100,bias=True)
    self.de5 = nn.Linear(100,200,bias=True)
    self.de6= nn.Linear(200,200,bias=True)
    self.de7 = nn.Linear(200,4,bias=True)
    self.tanh=nn.Tanh()
   
  def encoder(self,x):
    x=self.tanh(self.en1(x)).float()
    x=self.tanh(self.en2(x)).float()
    x=self.tanh(self.en3(x)).float()
    x=self.tanh(self.en4(x)).float()
    x=self.tanh(self.en5(x)).float()
    x=self.tanh(self.en6(x)).float()
    return (self.en7(x)).float()
  
  def decoder(self,z):
    z=self.tanh(self.de1(z)).float()
    z=self.tanh(self.de2(z)).float()
    z=self.tanh(self.de3(z)).float()
    z=self.tanh(self.de4(z)).float()
    z=self.tanh(self.de4(z)).float()
    z=self.tanh(self.de5(z)).float()
    z=self.tanh(self.de6(z)).float()
    return (self.de7(z)).float()

  def forward(self,x):
    a=self.encoder(x).float()
    a=self.decoder(a).float()
    return a.float(),encoder(x).float()
  def describe(self):
    print("input,200,200,100,100,50,25,3,25,100,100,200,200,output")
 
