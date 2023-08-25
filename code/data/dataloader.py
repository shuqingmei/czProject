import pandas as pd
import torch.utils.data as Data
import numpy as np
from  ..utils import tools
from  ..utils import timefeature
import random

class DatasetLoader(Data.Dataset):
    def __init__(self, seq_len, label_len, pred_len, flag=0) -> None:
        '''
        flag: 0=train, 1=test, 2=vaild
        output: (series_len, feature_num)
        '''
        super(DatasetLoader, self).__init__()
        df = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_max.csv')
        df_min = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_min.csv')
        df_mean = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_mean.csv')
        df_var = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_var.csv')
        df_max = df.iloc[:, 1:3].values
        df_min = df_min.iloc[:, 1:3].values
        df_var = df_var.iloc[:, 1:3].values
        df_mean = df_mean.iloc[:, 1:3].values
        df_data = np.concatenate((df_max, df_min, df_mean, df_var), axis=1)
        mean = np.average(df_data, axis=0)
        std = np.std(df_data, axis=0)
        df_stamp = df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # date列转换为日期时间格式
        self.data_stamp = timefeature.time_features(df_stamp, 0, 't')
        self.train_data = df_data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        l = int(len(self.train_data)* 0.8)
        # 训练集：测试集8：2

        self.train_data = (self.train_data - mean) / std 
        if flag==0:
            self.train_data = self.train_data[:l, :]
            self.data_stamp = self.data_stamp[:l, :]
            self.len = l - self.seq_len - self.pred_len
        elif flag==1: 
            self.len = len(self.train_data) - l - self.seq_len - self.pred_len
            self.train_data = self.train_data[l:, :]
            self.data_stamp = self.data_stamp[l:, :]
    
    def __len__(self):
        return 1000

    def __getitem__(self, index):
        index = random.randint(0, self.len - self.seq_len - self.pred_len)
        begin_en = index
        end_en = index + self.seq_len
        begin_de = index + self.seq_len - self.label_len
        end_de = begin_de + self.label_len + self.pred_len
        enc_in = self.train_data[begin_en: end_en, :]
        enc_mask = self.data_stamp[begin_en: end_en, :]
        dec_in = self.train_data[begin_de:end_de, :]
        dec_mask = self.data_stamp[begin_de:end_de, :]

        return enc_in, enc_mask, dec_in, dec_mask 
    
    def inverse_transform(self, data):
        return data * 2.55054430 + 3.60226484