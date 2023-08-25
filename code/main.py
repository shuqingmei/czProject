import pymysql
import sql
import predict
import pandas as pd
import numpy as np
import re

mysql_conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', db='windpredict')
# 建立与数据库的连接，指定了主机名、端口号、用户名、密码和数据库名等参数
try:
    cursor = mysql_conn.cursor()
    # 创建一个游标对象，该对象可以执行 SQL 查询和操作
except Exception as e:
    print(e)
    mysql_conn.rollback()

df = pd.read_csv('../dataset/xmb/项目部超声波风速风向_10T_max.csv')
pred_data = df[0: 720]
# 从一个 CSV 文件中读取数据，并将前720行保存在pred_data变量
checkpoint_path = '../log/checkpoint/Informer/20.pth'
# 指定了一个检查点文件的路径checkpoint_path
p, h = predict.eval(pred_data, checkpoint_path)
# 调用predict.eval(pred_data, checkpoint_path)函数对pred_data进行预测，返回的结果保存在变量p和h中
wind_max = np.max(p)
# 计算p中的最大值，将结果保存在wind_max变量中

grid = f"0, 0, 0, 0, 'no'"
# 点信息
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
# 预测时间
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
# 预测时间和数据 24*10=240分钟，240/60=4，所以是第四个小时
# sql.insert_gridinfo(mysql_conn, cursor, grid)
# sql.insert_datecalculate(mysql_conn, cursor, dateinfo)
sql.insert_hourcalculate(mysql_conn, cursor, hourinfo)
# 将给定的数据插入到 MySQL 数据库中的特定表格中