import predict
import pandas as pd

df = pd.read_csv('../dataset/xmb/项目部超声波风速风向_10T_max.csv')
pred_data = df[0: 720]
checkpoint_path = '../log/checkpoint/Informer/20.pth'
p, h = predict.eval(pred_data, checkpoint_path)
print(h)
print(p)
