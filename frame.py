import numpy as np
import pandas as pd


class Frame(object):
    def __init__(self,mu,FR,BW):
        # PRB2.JPG at Google drive- Mydrive- Master Program- Thesis- PRB
        # First raw  shows Bandwidtht in MHz (channel)
        # the number of raw shows the miu
        #each number shows maximum number PRB at specified numerology

        self.df2 = pd.read_excel(r'D:\Autonomous Systems\KTH\Thesis\Implementation\Data\Max.xlsx')
        # same file as the aformentioned but regarding to carrier frequency above 24 GHz
        self.df3 = pd.read_excel(r'D:\Autonomous Systems\KTH\Thesis\Implementation\Data\Max2.xlsx')
        self.mu=mu
        # if FR is true, the carrier frequncy is below 6GHz otherwise above 24 GHz
        self.FR=FR
        self.BW=BW
        # SCS in kHZ
        self.subcarSpace = 15 * np.power(2, mu)
        self.Tslot = 1/ np.power(2, mu)
        self.numslot_frame=10 * np.power(2, mu)
    def Maxnprb(self): #gives the maximum number of PRB in each numorology
        if self.FR == True: #below 6GHZ

            Max= self.df2[self.BW][self.mu]
        else: #above 24 GHz

            Max= self.df3[self.BW][self.mu]


        return Max

# p = Frame(2,True, 20)
# print(p.numslot_frame)
# print(p.BW)