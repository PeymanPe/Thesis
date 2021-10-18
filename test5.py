import numpy as np
import math
import matplotlib.pyplot as plt
from frame import Frame
from Delay2 import Ceq
from Delay2 import Dtot
# from debug import Ceq
# from debug import Dtot

import pandas as pd


# C_i values is calculated here based on specified senario


df = pd.read_excel(r'D:\Autonomous Systems\KTH\Thesis\New simulation\Data\table2ref.xlsx')
dff = df.values
pd.DataFrame(dff).to_numpy()

#Table V values (SISO, 20MHz, 6 BPS/HZ, 64-QAM
refvalue = np.array([20,6,1,6,1])


#packet size (unit: Bytes)
packetSize = 1500
#packet size (unit: bits)
packetsize = packetSize * 8
# file size(each user) 100kb
Lf =0.1 *8
# modulation index =4 16-QAM
nmod=4
#Switch bit rate (unit: Mbps)
SwitchBitRate = 800
#number of symbols per slot
nsymbol=14
#number of resources elements per PRB
nre = 12*nsymbol
#splitting II_D
split = 9
# 2*2 MIMO
Nant=2
# perctage of usage of resource blocks
nn=1
# number of bits per symble 16-QAM
nmod = 4

#Number of CPUs per Cloud Server
NCPU=2
#Base processor frequency
f=2.3
#Number of cores per CPU
Ncores=18
#Number of instructions per CPU cycle
Nipc=16

#assumption 20Mhz channel miu=0 , FR1
p = Frame(0,True, 20)

#T slot unit  =>  ms
# print(p.Tslot)


#GOPs capacity at each processing unit
ceq_CC = Ceq(NCPU,f,Ncores,Nipc)
ceq_RU = 300






#we assume 80


#Number of users
user = 5

#percentage of BW allocation for a service
sp=1


#number of allocated subcarriers
n_subcar = math.floor(sp * 12 * p.Maxnprb())

# BW = n_subcar * p.subcarSpace



def DelayChangeBW(p, n_subcar,C_percent):
    # number of radio units
    ru = 4
    BW = (n_subcar * p.subcarSpace)/1000
    n_subcar_user = math.floor(n_subcar/user)
    BW_user = (n_subcar_user * p.subcarSpace)/1000



    actualvalue = np.array([BW, nmod, 2, 6, 1])

    actToRef = actualvalue / refvalue

    actualvalue_user = np.array([BW_user, nmod, 2, 6, 1])
    # actToRefUser = actToRef.copy()
    # we assume we allocate equally to each user
    actToRefUser = actualvalue_user / refvalue

    dff2 = dff[:, 1:]

    # cj = dff[:, 0]
    cj2 = dff[:, 0]
    cj = np.copy(cj2)
    # print(cj)
    cjj = np.ones(16)
    # print(cjj)

    for i in range(16):
        for j in range(5):
            if i < 11:
                cjj[i] *= pow(actToRef[j], dff2[i, j])
            else:
                cjj[i] *= pow(actToRefUser[j], dff2[i, j])
    cj *= cjj
    # GOPs capacity allocated for CP and UP (first eleement for CP and the second for UP)
    if split == 9:
        # frac_CC = (np.sum(cj[-5:])*user*ru)/(np.sum(cj[-5:])*user*ru+cj[-6])
        frac_CC = 0.5
        ceq_CC2 = np.array([1-frac_CC, frac_CC])*ceq_CC * C_percent
        ceq_RU2 = np.array([1, 0]) * ceq_RU * C_percent
    elif split == 11:
        frac_RU = (cj[-5]*user)/(cj[-5]*user+ np.sum(cj[:-5]))
        ceq_RU2 = np.array([1-frac_RU, frac_RU])*ceq_RU * C_percent
        ceq_CC2 = np.array([0, 1]) * ceq_CC * C_percent

    dd = Dtot(Lf, packetsize, SwitchBitRate, n_subcar, nre, nmod, p.Tslot, cj, ceq_RU2, ceq_CC2, split, Nant, nn, nsymbol,
              user,ru)
    # Dtot(Lf, packetsize, SwitchBitRate, Nsc, nre, nmod, Tslot, cj, cRUEq, cCCEq, split, Nant, nn, nsymbol, user)
    return dd[0], dd[1], dd[2], dd[3], dd[4], cj

#percentage of BW allocation for a service
sp=1

n_subcar_max = 12 * p.Maxnprb()

# print(n_subcar_max)

# y = DelayChangeBW(p,n_subcar_max)[:-1]
# print("y")
# print(y)




# print(DelayChangeBW(p,1))
#what percent of DUs allocated for a service
x1 = np.linspace(0.2, 1, num=10)
# x1 = np.arange(5, n_subcar_max)
# x1 = np.arange(50, n_subcar_max, 20)


y1 = np.empty([len(x1),5])
for i in range(len(x1)):
    y1[i,:] = DelayChangeBW(p,n_subcar_max,x1[i])[:-1]
    # print(x1[i])



plt.plot(x1,y1[:,3], 'o-r')
plt.plot(x1,y1[:,2], 'o-g')
plt.plot(x1,y1[:,4], 'o-c')
plt.title('Delay components with varying allocated digital units')
plt.xlabel("percentage of allocated DUs")
plt.ylabel("Total delay")
plt.legend(('total delay', 'processing delay','Delay in RAN'), loc='upper right', shadow=True)

plt.grid()
plt.show()