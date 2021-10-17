import numpy as np
import matplotlib.pyplot as plt
from frame import Frame


# spliting point (
split = 8

#packet size (unit: Bytes)
packetSize = 1500

#packet size (unit: bits)
packetsize = packetSize * 8

#Midhaul Bit rate (unit: Mbps)
MHbitRate = 933

#Switch bit rate (unit: Mbps)
SwitchBitRate = 800

#number of resources elements per PRB
nre = 12*7

#modulation index
nmod = 4

#proppagation delay (unit in micro seconds)
Dp1 = 5
Dp2 = 5
DRtx = 1

#number of switches in series between central cloud and edge cloud
Nsw = 1

#queuing delay for simplicity we assume it is constant (unit micro  seconds)
Dq = 10

#number of radio units
r = 4

#power dependent component in equation 12
Pstx = 23

#fabric delay (unit in micro second)
Df = 5





# required bitrate for fronthaul
def MHbitRate(Nsc, Ts,nn,split,Nant,Nbits):
    if split ==9:
        MH = Nsc * 0.9 * 2 * Nbits * Nant * nn / Ts
    elif split ==0:
        MH = fs * 2 * Nbits * Nant
    elif split ==15:
        MH = 151
    return MH

def Ceq(NCPU,f,Ncores,Nipc):
    ceq1 = NCPU * f * Ncores * Nipc
    return ceq1

def Dtot(Lf, packetsize, SwitchBitRate, nprb, nre, nmod, Tslot, cj, cRUEq, cCCEq, split,Nant,nn,nsymbol,user):



    # switching delay (unit:microsecond)
    Dse = packetsize / SwitchBitRate

    # required bitrate for fronthaul
    Nsc = nprb*12
    #sample duration time
    Ts = Tslot/nsymbol
    MHbitRate1 = MHbitRate(Nsc, Ts, nn, split,Nant,nmod)



    # Transmission delay (unit:microsecond)
    DMtx = Lf / MHbitRate1

    nprb2 = nprb.copy()

    nprb2/=user

    # Equation 5 (unitless)
    Nslot = Lf / (nprb2 * nre * nmod)

    # equation 4 (unit:milisecond)
    DRtx = Nslot * Tslot

    #equation 23 (unit giga operation per radio subframe
    #we have 16 total number of CP and UP functions
    if split == 0:
        C_i_EC = 0
        C_i_CC = np.sum(cj)
        n2 = 0
        n1 = 1
    elif split == 16:
        C_i_EC = np.sum(cj)
        C_i_CC = 0
        n1 = 0
        n2 = 1
    else :
        C_i_EC = np.sum(cj[:split])
        C_i_CC = np.sum(cj[split:])
        n1 = 1
        n2 = 1





    # Equation 7 (unit miliseconds)
    dECpr = C_i_EC / cRUEq
    dCCpr = C_i_CC / cCCEq

    # Equation 6 (unit miliseconds)
    Dpr = n1 * dCCpr + n2 * dECpr

    # equation 1 (unit milisecond)
    Dtot =  DMtx * 0.001 +  Dp1 * 0.001 +  DRtx + Dpr +  Nsw * (Dq * 0.001 + Df * 0.001 + Dse * 0.001) + \
             Dp2 * 0.001

    return dECpr, dCCpr, Dpr , Dtot, DRtx


# file size 100kb
Lf =100*8

#
p = Frame(2,False,20)
