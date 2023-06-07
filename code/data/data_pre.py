import numpy as np
import pandas as pd
from datetime import datetime

def date_re():
    csv1 = pd.read_csv(f'../dataset/项目部超声波风速风向_20220810.csv')
    csv2 = pd.read_csv(f'../dataset/项目部-超声波风速风向.csv')
    print('loaded')
    csv = pd.concat([csv1, csv2])
    # 有效列选择
    csv = pd.DataFrame(csv, columns=['col0', 'col1', 'col2'])
    csv['date'] = pd.to_datetime(csv['col0'], format="%Y-%m-%d-%H-%M-%S.%f")
    csv = pd.DataFrame(csv, columns=['date', 'col1', 'col2'])
    csv = csv.set_index('date')
    # 异常值处理
    csv.col1 = (csv.col1 < 360) * (csv.col1 >= 0) * csv.col1
    csv.col2 = (csv.col2 < 50) * (csv.col2 >= 0) * csv.col2
    # 空白时间填充
    a = csv.resample('10T').max().fillna(0)
    a.to_csv('../dataset/xmb/项目部超声波风速风向_10T_max.csv')
    a = csv.resample('10T').mean().fillna(0)
    a.to_csv('../dataset/xmb/项目部超声波风速风向_10T_mean.csv')
    a = csv.resample('10T').var().fillna(0)
    a.to_csv('../dataset/xmb/项目部超声波风速风向_10T_var.csv')
    a = csv.resample('10T').min().fillna(0)
    a.to_csv('../dataset/xmb/项目部超声波风速风向_10T_min.csv')