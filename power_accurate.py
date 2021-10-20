import numpy as np
import pandas as pd
import math
from frame import Frame
from Delay2 import Ceq
from Time_Slot import Timeslot
import matplotlib.pyplot as plt



powCCminDU = 10
powCCmaxDU = 1000

powRUminDU = 5
powRUmaxDU = 200


powCCcool = 5
powSWstatic =5
roSW = math.pow(10,-6)
#load independent power consumption at radio unit
powRU_indep=10
#required power to transmit 1 subcarrier =26dBm
powRU_tx = 0.398
#total power consumed


oo = pd.read_excel(r'D:\Autonomous Systems\KTH\Thesis\New simulation\Data\Lf.xlsx')
oo2=oo.groupby(['RU','serviceNo']).sum()

oo3=oo2.unstack('serviceNo')
# print(oo3)

# number of radio units
ru = oo.RU.value_counts().shape[0]

#number of service
s_num = oo.serviceNo.value_counts().shape[0]


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
    # # number of radio units
    # ru = 4
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
    oo_user2 = oo_user.loc['s' + str(s)]
    # Nsc_user = Nsc / oo_user2[ru]

    # Nslot = math.ceil(file3[ru]*1000 / (Nsc_user* nmod*7))






    oo2 = oo.groupby(['RU', 'serviceNo']).sum()

    oo3 = oo2.unstack('serviceNo')





    powRU = np.zeros(ru)
    powRU_DU = np.zeros(ru)

    UFru = np.zeros(ru)

    Timeslot2 = Timeslot(frame, Nsc, oo, nmod)

    data = Timeslot2.pandata_TimeSlot()


    for i in range(ru):
        C_RU_sum = np.zeros(2)
        for j in range(p.numslot_frame):

            data2 = data[(data.RU=='ru'+str(i+1)) & (data.serviceNo=='s'+str(s))]
            data3=np.array(data2.iloc[:,2:],dtype=int)[0]
            #We assume the number of allocated subcarriers per user is fixed
            #data2[j] is the number of users at slot number j
            NS= (Nsc[i,s-1]*data3[j])/data3[0]
            # print("at Ru {} at time slot {} Ns ={}".format(i+1,j+1,NS))

            if NS !=0:
                C_arr_new = Cj_calculate(p, NS, data3[j])
                C_RU_sum += np.array([C_arr_new[0], C_arr_new[1] * data3[j]])
                # print(C_arr_new)
                # print(C_RU_sum)



        # c_arr = Cj_calculate(p, Nsc[i,s-1], oo_user2[i])
        UFru[i] = (C_RU_sum[0] + C_RU_sum[1]) / (ceq_RU * C_percent * p.numslot_frame)

        powRU_DU[i] = powRUminDU + (powRUmaxDU - powRUminDU) * UFru[i]
        sum3 = Timeslot2.sum_num_user_slot(s, i+1)

        #I need update the equation in overleaf
        powRU[i] = powRU_indep/s_num + powRU_tx * Nsc[i,s-1] *(sum3/(p.numslot_frame * oo_user2[i]))+ powRU_DU[i]



    #power in switches
    Lftot = 0
    for i in range(oo3.shape[0]):
        for j in range(oo3.shape[1]):
            Lftot += oo3.loc['ru'+str(i+1)][j]


    c_arr2= np.zeros(2)

    for i in range(ru):
        for j in range(p.numslot_frame):

            data2 = data[(data.RU=='ru'+str(i+1)) & (data.serviceNo=='s'+str(s))]
            data3 = np.array(data2.iloc[:,2:],dtype=int)[0]
            NS= (Nsc[i,s-1]*data3[j])/data3[0]
            # print("at Ru {} at time slot {} Ns ={}".format(i + 1, j + 1, NS))

            if NS !=0:
                C_arr_new = Cj_calculate(p, NS, data3[j])
                c_arr2 += np.array([C_arr_new[2], C_arr_new[3] * data3[j]])
                # print(C_arr_new)
                # print(c_arr2)






    #power consumed for processiunf
    UFcc = (c_arr2[0] + c_arr2[1] ) / (ceq_CC * C_percent * p.numslot_frame)
    # UFcc = (cj_CC_CP * d_CC + cj_CC_UP * d_CC2) / (ceq_CC * 10 * C_percent)
    powCCDU = powCCminDU + (powCCmaxDU-powCCminDU)*UFcc

    powSW = powSWstatic/s_num + roSW * Lftot

    powCC = powSW + powCCDU + (powCCcool/s_num)


    powTot = powCC + np.sum(powRU)

    return powTot, powCC, powRU[0], powCCDU, powRU_DU[0]






#assumption 20Mhz channel miu=2 , FR1
frame = Frame(0,True, 20)
# modulation index =4 16-QAM
nmod=4


Nsc = 12 * frame.Maxnprb()
Nsc2 = np.ones((4,3)) * Nsc

C_percent = 1
s = 1


print(powerslice(frame,ru,oo,s,Nsc2,C_percent))










x1 = np.linspace(0.2, 1, num=10)



y1 = np.empty([len(x1),5])
for i in range(len(x1)):
    y1[i,:] = powerslice(frame, ru, oo, s, x1[i]*np.ones((4,3))* Nsc, C_percent)
    # y1[i,:] = DelayChangeBW(p,x1[i],C_percent)[:-1]
    # print(x1[i])

x2=np.array(["{:.0%}".format(i) for i in x1])

plt.plot(x2,y1[:,0], 'o-r')
plt.plot(x2,y1[:,1], 'o-g')
plt.plot(x2,y1[:,2], 'o-c')
plt.plot(x2,y1[:,3], 'o-y')
plt.plot(x2,y1[:,4], 'o-k')
plt.title('slice_Power components for varying allocated bandwidth')
plt.xlabel("Percentage of allocated BW for a slice")
plt.ylabel("Power (W)")
plt.legend(('total power', 'power at CC','power at RU','power for processing at CC','power for processing at RU'), loc='upper left', shadow=True)

plt.grid()
plt.show()