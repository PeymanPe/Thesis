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
p = Frame(0,True, 40)

#T slot unit  =>  ms
# print(p.Tslot)


#GOPs capacity at each processing unit
ceq_CC = Ceq(NCPU,f,Ncores,Nipc)
ceq_RU = 300


#Number of users
user = 5



def DelayChangeBW(p,sp):
    nprb = sp * p.Maxnprb()
    BW_max = p.BW


    BW = BW_max*sp

    actualvalue = np.array([BW, nmod, 2, 6, 1])

    actToRef = actualvalue / refvalue
    actToRefUser = actToRef.copy()
    # we assume we allocate equally to each user
    actToRefUser[0] /= user

    dff2 = dff[::, 1:]

    cj = dff[::, 0]
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

    dd = Dtot(Lf, packetsize, SwitchBitRate, nprb, nre, nmod, p.Tslot, cj, ceq_RU, ceq_CC, split, Nant, nn, nsymbol,
              user)
    return dd

#percentage of BW allocation for a service
sp=1


print(DelayChangeBW(p,1)[3])
x1 = np.linspace(0.1, 1, 10)
# x1 = np.linspace(0.1, 0.2)
y1 = np.zeros((len(x1),5))
for i in range(len(x1)):
    y1[i,::] = DelayChangeBW(p,x1[i])


plt.plot(x1,y1[::,3], 'o-r')
plt.plot(x1,y1[::,2], 'o-g')
plt.plot(x1,y1[::,4], 'o-c')
plt.title('Delay components with varying allocated bandwidth')
plt.xlabel("PRC usage percentage(BW)")
plt.ylabel("Total delay")
plt.legend(('total delay', 'processing delay','Delay in RAN'), loc='upper right', shadow=True)
# fig, ax = plt.subplots()
# ax.plot(x1, y1, 'o-')
# ax.set_xlabel('PRC usage percentage(BW)')  # Add an x-label to the axes.
# ax.set_ylabel('Total delay')  # Add a y-label to the axes.
plt.grid()
plt.show()

# print(len(x1))