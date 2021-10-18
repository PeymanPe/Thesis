import pandas as pd
import numpy as np

oo = pd.read_excel(r'D:\Autonomous Systems\KTH\Thesis\New simulation\Data\Lf.xlsx')
oo4 = oo[(oo.RU == "ru1")]
oo2=oo.groupby(['RU','serviceNo']).sum()

oo3=oo2.unstack('serviceNo')
pp=oo
pp["Delay"]= 1
pp2 =oo.groupby(['RU','serviceNo'])
for i , j in pp2:
    print(i)
    j["Delay"]=1
pp2.unstack(['RU','serviceNo'])

print(pp2)
