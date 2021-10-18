import numpy as np
import pandas as pd
# from frame import Frame
from Delay2 import nmod
import math









class Timeslot(object):
    def __init__(self,frame,Nsc_slice_share,oo, nmod):
        self.frame = frame
        self.Nsc_slice_share = Nsc_slice_share
        self.oo = oo
        self.ru = oo.RU.value_counts().shape[0]
        self.s = oo.serviceNo.value_counts().shape[0]
        self.nmod = nmod


    # maximum number of time slot in service number s for transmiting
    def numslut_max(self,s2):
        kk = self.oo.groupby(['serviceNo', 'RU']).agg('max')
        kk2 = kk.unstack()
        file3 = kk2.loc['s' + str(s2)]
        # print(file3)
        Nslot = np.zeros(self.ru)

        oo_user = self.oo.groupby(['serviceNo', 'RU']).size().unstack()
        oo_user2 = oo_user.loc['s' + str(s2)]
        # Nsc_user
        for i in range(self.ru):
            Nsc_user = math.floor(self.Nsc_slice_share / oo_user2[i])
            Nslot[i] = math.ceil(file3[i] * 1000 * 8 / (Nsc_user * self.nmod * 7))

        return Nslot

    # minmum number of time slot in service number s for transmiting
    def numslut_min(self,s2):
        kk = self.oo.groupby(['serviceNo', 'RU']).agg('min')
        kk2 = kk.unstack()
        file3 = kk2.loc['s' + str(s2)]
        # print(file3)
        Nslot = np.zeros(self.ru)

        oo_user = self.oo.groupby(['serviceNo', 'RU']).size().unstack()
        oo_user2 = oo_user.loc['s' + str(s2)]
        # Nsc_user
        for i in range(self.ru):
            Nsc_user = math.floor(self.Nsc_slice_share / oo_user2[i])
            Nslot[i] = math.ceil(file3[i] * 1000 * 8 / (Nsc_user * self.nmod * 7))

        return Nslot

    def num_user_inTimeSlot(self, s2, ru2):

        oo_user = self.oo.groupby(['serviceNo', 'RU']).size().unstack()
        oo_user2 = oo_user.loc['s' + str(s2)]

        ## oo_new has a colum showing the number of requires time slot



        oo_new = self.oo.copy()
        oo_new['num_user'] = oo_new.apply(lambda x: oo_user.loc[x['serviceNo']][x['RU']], axis=1)
        oo_new['ser'] = oo_new.apply(lambda x: int(x['serviceNo'].replace('s', '')), axis=1)
        oo_new['subcar_user'] = oo_new.apply(lambda x: math.floor(self.Nsc_slice_share / x['num_user']), axis=1)


        # oo_new['subcar_user'] = oo_new.apply(lambda x: math.floor(NscS[x['ser']] / x['num_user']), axis=1)
        oo_new['num_slot'] = oo_new.apply(lambda x: math.ceil(x['FileSize'] * 1000 * 8 / (x['subcar_user'] * self.nmod * 7)),
                                          axis=1)

        oo_new2 = oo_new.groupby(['serviceNo', 'RU', 'num_slot']).size()

        jj = oo_new2.loc['s' + str(s2), 'ru' + str(ru2)].index
        jj2 = np.sort(jj)
        jj3 = jj2[::-1]

        # num_slot_s_ru = np.zeros(self.numslut_max(s2,nmod).astype(int)[ru2 - 1])
        num_slot_s_ru = np.zeros(self.frame.numslot_frame)

        for i in range(len(num_slot_s_ru)):
            if i < self.numslut_min(s2)[ru2 - 1]:
                oo_select = oo_new[(oo_new.serviceNo == 's' + str(s2)) & (oo_new.RU == 'ru' + str(ru2))]
                num_slot_s_ru[i] = oo_select.shape[0]
            else:
                oo_select = oo_new[
                    (oo_new.serviceNo == 's' + str(s2)) & (oo_new.RU == 'ru' + str(ru2)) & (oo_new.num_slot > i)]
                num_slot_s_ru[i] = oo_select.shape[0]

        # print(num_slot_s_ru)
        return num_slot_s_ru

    def pandata_TimeSlot(self):
        tot = self.s * self.ru
        RU_list = []
        serviceNo_list = []
        for i in range(self.s):
            for j in range(self.ru):
                RU_list.append('ru'+ str(j+1))
                serviceNo_list.append('s'+ str(i+1))

        scores = {'RU': RU_list,
                  'serviceNo': serviceNo_list,
                  # "num_user_inTimeSlot": [75, 92, 94]
                  }
        df = pd.DataFrame(scores)
        for i in range(self.frame.numslot_frame):
            df['slot' + str(i + 1)] = df.apply(lambda x: self.num_user_inTimeSlot(int(x['serviceNo'].replace('s', '')), int(x['RU'].replace('ru', '')))[i],axis=1)
            # df['slot' + str(i + 1)] = df.apply(lambda x: i,axis=1)
            # df['slot' + str(i + 1)] = i

        return df





# oo = pd.read_excel(r'D:\Autonomous Systems\KTH\Thesis\New simulation\Data\Lf.xlsx')
# frame = Frame(0,True, 20)
# Nsc = 12 * frame.Maxnprb()
#
# Timeslot2 = Timeslot(frame,Nsc,oo)
# print(Timeslot2.numslut_max(2,4))
#
#
# s2 = 1
# ru2 = 1
# print(Timeslot2.num_user_inTimeSlot(s2,ru2))
# print(Timeslot2.pandata_TimeSlot())