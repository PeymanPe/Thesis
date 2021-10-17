import numpy as np
import math
import matplotlib.pyplot as plt
from frame import Frame
from Delay import Ceq
from Delay import Dtot
from Delay import MHbitRate
from C_i import dff

#Table V values (SISO, 20MHz, 6 BPS/HZ, 64-QAM
refvalue = np.array([20,6,1,6,1])


#packet size (unit: Bytes)
packetSize = 1500
#packet size (unit: bits)
packetsize = packetSize * 8
# file size(each user) 100kb
Lf =100*8
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

#assumption 20Mhz channel miu=2 , FR1
p = Frame(2,True, 20)

#T slot unit  =>  ms
# print(p.Tslot)
#percentage of BW allocation for a service
sp=1

nprb = sp * p.Maxnprb()
#GOPs capacity at each processing unit
ceq_CC = Ceq(NCPU,f,Ncores,Nipc)
ceq_RU = 300

# print(dff)
BW = p.BW
# print(BW)

#Number of users
user = 5

actualvalue = np.array([BW,nmod,2,6,1])

actToRef = actualvalue/refvalue
actToRefUser = actToRef.copy()
#we assume we allocate equally to each user
actToRefUser[0]/=user



dff2 = dff[::,1:]

cj = dff[::,0]
print(cj)
cjj =np.ones(16)
# print(cjj)

for i in range(16):
    for j in range(5):
        if i<12:
            cjj[i] *= pow(actToRef[j],dff2[i,j])
        else:
            cjj[i] *= pow(actToRefUser[j], dff2[i, j])
cj *= cjj





dd=Dtot(Lf, packetsize, SwitchBitRate, nprb, nre, nmod, p.Tslot, cj, ceq_RU, ceq_CC, split, Nant, nn,nsymbol,user)


print(dd)

