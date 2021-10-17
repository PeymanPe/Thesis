import pandas as pd
import numpy as np

from frame import Frame

# C_i values is calculated here based on specified senario


df = pd.read_excel(r'D:\Autonomous Systems\KTH\Thesis\New simulation\Data\table2ref.xlsx')
dff = df.values
pd.DataFrame(dff).to_numpy()


# print(dff.ndim)
# print(dff)
# print(dff[:, 1:])
# print(dff[::, 1:])
