import pymysql
import sql
import predict
import pandas as pd
import numpy as np
import re

mysql_conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', db='windpredict')
try:
    cursor = mysql_conn.cursor()
except Exception as e:
    print(e)
    mysql_conn.rollback()

df = pd.read_csv('../dataset/xmb/项目部超声波风速风向_10T_max.csv')
pred_data = df[0: 720]
checkpoint_path = '../log/checkpoint/Informer/20.pth'
p, h = predict.eval(pred_data, checkpoint_path)
wind_max = np.max(p)


grid = f"0, 0, 0, 0, 'no'"
dateinfo = f'''
'{h}',
-99, -99,
-99, -99,  
-99, -99,  
-99, -99,  
-99, -99,  
-99, -99,  
-99, -99,  
'''
hourinfo = f'''
0, '{h}',
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
{wind_max}, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99,
-99, -99, -99
'''.replace("\n", '')
# sql.insert_gridinfo(mysql_conn, cursor, grid)
# sql.insert_datecalculate(mysql_conn, cursor, dateinfo)
sql.insert_hourcalculate(mysql_conn, cursor, hourinfo)
