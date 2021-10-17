import numpy as np
import pandas as pd
import math
from frame import Frame
from Delay2 import Ceq
import matplotlib.pyplot as plt


# number of radio units
ru = 3

powCCminDU = 50
powCCmaxDU = 1000

powRUminDU = 50
powRUmaxDU = 100


powCCcool = 5
powSWstatic =15
roSW = 0.000001
#load independent power consumption at radio unit
powRU_indep=10
#required power to transmit 1 subcarrier = 26dBM
powRU_tx = 0.39
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


# #assumption 20Mhz channel miu=0 , FR1
# p = Frame(0,True, 20)

# modulation index =4 16-QAM
nmod=4

#Table V values (SISO, 20MHz, 6 BPS/HZ, 64-QAM
refvalue = np.array([20,6,1,6,1])

# C_i values is calculated here based on specified senario


df = pd.read_excel(r'D:\Autonomous Systems\KTH\Thesis\New simulation\Data\table2ref.xlsx')
dff = df.values
pd.DataFrame(dff).to_numpy()

#splitting II_D
split = 9



# we only assume we have one switch



def Cj_calculate(p, n_subcar,user):
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
    UFru = np.zeros(ru)
    # UFru = np.empty([1, ru])



    for i in range(ru):
        c_arr = Cj_calculate(p, Nsc, oo_user2[i])
        print("c_arr is {} at radio unit {}".format(c_arr,i+1))


        UFru[i] = (c_arr[0]  + c_arr[1] * oo_user2[i])/(ceq_RU * C_percent)
        print("cj_RU_CP = {} and ceq_RU = {}".format(c_arr[0], ceq_RU))
        print("UFru at radio unit {} is {}".format(i+1,UFru[i]))
        powRU_DU[i] = powRUminDU + (powRUmaxDU - powRUminDU) * UFru[i]
        powRU[i] = powRU_indep + powRU_tx * Nsc + powRU_DU[i]

    #power in switches
    Lftot = 0
    for i in range(oo3.shape[0]):
        for j in range(oo3.shape[1]):
            Lftot += oo3.loc['ru'+str(i+1)][j]

    c_arr2= np.zeros(2)

    for i in range(ru):
        c_arr2 += Cj_calculate(p, Nsc, oo_user2[i])[2:]



    #power consumed for processiunf
    UFcc = (c_arr2[0]  + c_arr2[1] * oo_user2.sum()) / (ceq_CC * C_percent)
    powCCDU = powCCminDU + (powCCmaxDU-powCCminDU)*UFcc

    powSW = powSWstatic/s + roSW * Lftot


    powCC = powSW + powCCDU + powCCcool
    # powCC = powCCDU + powCCcool


    powTot = powCC + np.sum(powRU)
    # UFru[i]
    return powTot, powCC, powRU[0] , powCCDU, powRU_DU[0]
    # return UFru[0], powCC, powRU[0] , powCCDU, powRU_DU[0]


#assumption 20Mhz channel miu=2 , FR1
p = Frame(0,True, 20)
Nsc = 12 * p.Maxnprb()
n_subcar_max = 12 * p.Maxnprb()

C_percent = 1
s = 1
# UFru=np.array([1,1,1,1])
# UFcc=1

# print(powerslice(p,ru,oo,s,Nsc,C_percent))




x1 = np.arange(50, n_subcar_max, 20)


y1 = np.empty([len(x1),5])
for i in range(len(x1)):
    y1[i,:] = powerslice(p, ru, oo, s, x1[i], C_percent)
    # y1[i,:] = DelayChangeBW(p,x1[i],C_percent)[:-1]
    # print(x1[i])



plt.plot(x1,y1[:,0], 'o-r')
plt.plot(x1,y1[:,1], 'o-g')
plt.plot(x1,y1[:,2], 'o-c')
plt.plot(x1,y1[:,3], 'o-y')
plt.plot(x1,y1[:,4], 'o-k')
plt.title('Power components for varying allocated bandwidth')
plt.xlabel("number of subcarriers")
plt.ylabel("Power")
plt.legend(('total power', 'power at CC','power at RU','power for processing at CC','power for processing at RU'), loc='upper right', shadow=True)

plt.grid()
plt.show()










