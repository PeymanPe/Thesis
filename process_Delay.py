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

# Table V values (SISO, 20MHz, 6 BPS/HZ, 64-QAM
refvalue = np.array([20, 6, 1, 6, 1])

# packet size (unit: Bytes)
packetSize = 1500
# packet size (unit: bits)
packetsize = packetSize * 8
# file size(each user) 100kb
Lf = 0.5 * 8
# modulation index =4 16-QAM
nmod = 4
# Switch bit rate (unit: Mbps)
SwitchBitRate = 800
# number of symbols per slot
nsymbol = 14
# number of resources elements per PRB
nre = 12 * nsymbol
# splitting II_D
split = 9
# 2*2 MIMO
Nant = 2
# perctage of usage of resource blocks
nn = 1
# number of bits per symble 16-QAM
nmod = 4

# Number of CPUs per Cloud Server
NCPU = 2
# Base processor frequency
f = 2.3
# Number of cores per CPU
Ncores = 18
# Number of instructions per CPU cycle
Nipc = 16



# T slot unit  =>  ms
# print(p.Tslot)


# GOPs capacity at each processing unit
ceq_CC = Ceq(NCPU, f, Ncores, Nipc)
ceq_RU = 300

# we assume 80


# Number of users
user = 5

# percentage of BW allocation for a service
sp = 1

# number of allocated subcarriers
# n_subcar = math.floor(sp * 12 * p.Maxnprb())


# BW = n_subcar * p.subcarSpace


def DelayChangeBW(p, n_subcar, C_percent, Lf2):
    # number of radio units
    ru = 4
    BW = (n_subcar * p.subcarSpace) / 1000
    n_subcar_user = math.floor(n_subcar / user)
    BW_user = (n_subcar_user * p.subcarSpace) / 1000

    actualvalue = np.array([BW, nmod, 1, 6, 1])

    actToRef = actualvalue / refvalue

    actualvalue_user = np.array([BW_user, nmod, 1, 6, 1])
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
        ceq_CC2 = np.array([1 - frac_CC, frac_CC]) * ceq_CC
        ceq_RU2 = np.array([1, 0]) * ceq_RU * C_percent

    elif split == 11:
        frac_RU = (cj[-5] * user) / (cj[-5] * user + np.sum(cj[:-5]))
        ceq_RU2 = np.array([1 - frac_RU, frac_RU]) * ceq_RU
        ceq_CC2 = np.array([0, 1]) * ceq_CC * C_percent

    dd = Dtot(Lf2, packetsize, SwitchBitRate, n_subcar, nre, nmod, p.Tslot, cj, ceq_RU2, ceq_CC2, split, Nant, nn,
              nsymbol,
              user, ru)
    # Dtot(Lf, packetsize, SwitchBitRate, Nsc, nre, nmod, Tslot, cj, cRUEq, cCCEq, split, Nant, nn, nsymbol, user)
    return dd[0], dd[1], dd[2]


# percentage of BW allocation for a service
sp = 1

# n_subcar_max = 12 * p.Maxnprb()




# # what percent of DUs allocated for a service
# C_percent = 0.1
# # x1 = np.arange(5, n_subcar_max)
# x1 = np.arange(50, n_subcar_max, 20)
#
# y1 = np.empty([len(x1), 3])
# for i in range(len(x1)):
#     y1[i, :] = DelayChangeBW(p, x1[i], C_percent, Lf)
#     # print(x1[i])
#
#
#
#
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# ax.plot(x1, y1[:, 0], 'o-r')
# ax.plot(x1, y1[:, 1], 'o-g')
# ax.plot(x1, y1[:, 2], 'o-c')
# ax.axhline(y=1, color='r', linestyle='-')
# ax.text(100, 1.25, 'processing time threshold', fontsize=8, color='r')
#
# ax.set_title(
#     "Processing Delay components with varying allocated bandwidth, miu=0  with slice containing 5 users when 10% of total computing capacity is allocated ")
#
# ax.set(xlabel='number of subcarriers allocated for a slice', ylabel='Total delay (ms)')
# ax.legend(('processing delay at CC', 'processing delay at RU', 'total processing delay'), loc='upper right', shadow=True)
#
# minor_ticks = np.arange(0, 2, 0.1)
# major_ticks = np.arange(0, 2, 0.5)
# ax.set_yticks(minor_ticks, minor=True)
# ax.set_yticks(major_ticks)
#
# # ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)
#
# plt.show()





#threshold computing

mu2 =np.array([0,1,2])
# assumption 20Mhz channel miu=0 , FR1
# p = Frame(mu2[0], True, 20)
# n_subcar_max = 12 * p.Maxnprb()

# what percent of DUs allocated for a service
C_percent = 0.1
# x1 = np.arange(5, n_subcar_max)
x1 = np.linspace(0.1, 1, num=10)
# x1 = np.arange(50, n_subcar_max, 20)
C_percent = np.linspace(0.01, 0.2, num=30)

y1 = np.empty([len(x1), len(C_percent), len(mu2)])

for i in range(len(x1)):
    for j in range(len(C_percent)):
        for k in range(len(mu2)):
            p = Frame(mu2[k], True, 20)
            n_subcar_max = 12 * p.Maxnprb()
            y1[i,j,k] = DelayChangeBW(p, math.floor(x1[i] * n_subcar_max), C_percent[j], Lf)[2]

# y2 = np.zeros(len(x1))
y2 = np.empty([len(x1),len(mu2)])

for i in range(len(x1)):
    for j in range(len(C_percent)):
        for k in range(len(mu2)):
            p = Frame(mu2[k], True, 20)
            if y1[i,j,k] < p.Tslot:
                break
            else:
                y2[i,k]=C_percent[j]


# print(y2)
# print(C_percent)

x2=np.array(["{:.0%}".format(i) for i in x1])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(x2, y2[:,0], 'o-r')
ax.plot(x2, y2[:,1], 'o-g')
ax.plot(x2, y2[:,2], 'o-b')
# ax.axhline(y=1, color='r', linestyle='-')
# ax.text(100, 1.25, 'processing time threshold', fontsize=8, color='r')

ax.set_title(
    "Minimum percentage for processing with different numerology")

ax.set(xlabel='number of subcarriers allocated for a slice', ylabel='Threshold percentage for processing allocation percentage')
ax.legend(('miu=0', 'miu=1', 'miu=2'), loc='upper right', shadow=True)

minor_ticks = np.arange(0, 0.2, 0.01)
major_ticks = np.arange(0, 0.2, 0.04)
ax.set_yticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)


ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

plt.show()