import numpy as np
import pandas as pd
import math
from frame import Frame
from Delay2 import Ceq


# number of radio units
ru = 3

powCCminDU = 10
powCCmaxDU = 1000

powRUminDU = 5
powRUmaxDU = 200


powCCcool = 5
powSWstatic =5
roSW = 5
#load independent power consumption at radio unit
powRU_indep=10
#required power to transmit 1 subcarrier
powRU_tx = 10
#total power consumed


oo = pd.read_excel(r'D:\Autonomous Systems\KTH\Thesis\New simulation\Data\Lf.xlsx')
oo2=oo.groupby(['RU','serviceNo']).sum()

oo3=oo2.unstack('serviceNo')
print(oo3)

#Number of CPUs per Cloud Server
NCPU=2
#Base processor frequency
f=2.3
#Number of cores per CPU
Ncores=18
#Number of instructions per CPU cycle
Nipc=16



#GOPs capacity at each processing unit
ceq_CC = Ceq(NCPU,f,Ncores,Nipc)
ceq_RU = 300


#assumption 20Mhz channel miu=0 , FR1
p = Frame(0,True, 20)



# we only assume we have one switch



def Cj_calculate(p, n_subcar,C_percent):
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

        cj_CC_UP = np.sum(cj[-5:])
        cj_CC_CP = cj[-6]
        cj_RU_UP = 0
        cj_RU_CP = np.sum(cj[:-6])


    elif split == 11:
        cj_CC_UP = np.sum(cj[-4:])
        cj_CC_CP = 0

        cj_RU_UP = cj[-5]
        cj_RU_CP = np.sum(cj[:-5])


    c_arr = np.array([cj_RU_CP, cj_RU_UP, cj_CC_CP, cj_CC_UP])


    return c_arr



#Slice power (s is the service number, ru is the number of RUs, Nsc is the allocated subcarrier for that service
#C_percent is the allocated bandwidth
def powerslice(p,ru,oo,s,Nsc,C_percent):
    oo_s = oo[oo.serviceNo == ("s" + str(s))]


    kk = oo.groupby(['serviceNo','RU']).agg('max')
    kk2=kk.unstack()
    file3=kk2.loc['s'+ str(s)]

    oo_user = oo.groupby(['serviceNo','RU']).size().unstack()
    oo_user2 = oo_user.loc['s1']
    Nsc_user = Nsc / oo_user2[ru]

    Nslot = math.ceil(file3[ru]*1000 / (Nsc_user* nmod*7))






    oo2 = oo.groupby(['RU', 'serviceNo']).sum()

    oo3 = oo2.unstack('serviceNo')





    powRU = np.zeros(ru)
    powRU_DU = np.zeros(ru)



    for i in range(ru):

        UFru[i] = (cj_RU_CP * d_RU + cj_RU_UP * d_RU2)/(ceq_RU * 10 * C_percent)
        powRU_DU[i] = powRUminDU + (powRUmaxDU - powRUminDU) * UFru[i]
        powRU[i]=powRU_indep + powRU_tx * Nsc + powRU_DU[i]

    #power in switches
    Lftot = 0
    for i in range(oo3.shape[0]):
        for j in range(oo3.shape[1]):
            Lftot += oo3.loc['ru'+str(i+1)][j]


    #power consumed for processiunf
    UFcc = (cj_CC_CP * d_CC + cj_CC_UP * d_CC2) / (ceq_CC * 10 * C_percent)
    powCCDU = powCCminDU + (powCCmaxDU-powCCminDU)*UFcc

    powSW = powSWstatic/s + roSW * Lftot

    powCC = powSW+ powCCDU + powCCcool

    powTot = powCC + np.sum(powRU)

    return powTot, powCC, powRU[0]


#assumption 20Mhz channel miu=2 , FR1
p = Frame(2,True, 20)

s = 1
sp=80
UFru=np.array([1,1,1,1])
UFcc=1

print(powerslice(p,ru,oo,s,sp,UFru,UFcc))
