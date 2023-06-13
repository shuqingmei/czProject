import torch
import model.Informer as net
import pandas as pd
import utils.timefeature as timef
import numpy as np

def eval(pred_data, checkpoint_path):
    
    device = torch.device('cuda:0')
    seq_len = 360
    pred_len = 24
    label_len = 180
    input_size = 8
    output_size = 1
    mean = 3.60226484
    std = 2.55054430
    freq = '10min'
    pred_data['date'] = pd.to_datetime(pred_data.date)
    enc_mask = timef.time_features(pred_data, 0, 't')   
    enc_mask = torch.from_numpy(enc_mask).float().to(device).unsqueeze(dim=0)
    pred_date = pd.date_range(pred_data.date.values[-1], periods=pred_len+1, freq=freq)
    hour = pred_date[-1]
    df_stamp = pd.DataFrame(columns = ['date'])
    df_stamp.date = list(pred_data.date.values[-label_len:]) + list(pred_date[1:])
    dec_mask = timef.time_features(df_stamp, 0, 't')
    dec_mask = torch.from_numpy(dec_mask).float().to(device).unsqueeze(dim=0)

    model = net.Informer(input_size, input_size, out_len = pred_len, c_out=output_size, freq='t')
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])  

    df = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_max.csv')
    df_min = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_min.csv')
    df_mean = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_mean.csv')
    df_var = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_var.csv')
    df_max = df.iloc[:, 1:3].values
    df_min = df_min.iloc[:, 1:3].values
    df_var = df_var.iloc[:, 1:3].values
    df_mean = df_mean.iloc[:, 1:3].values
    pred_data = np.concatenate((df_max, df_min, df_mean, df_var), axis=1)[0:360, :]
    mean = np.average(pred_data, axis=0)
    std = np.std(pred_data, axis=0)
    print(pred_data[:, 1])
    enc_in = (pred_data -mean) /std
    enc_in = torch.from_numpy(enc_in).float().to(device)
    enc_in = enc_in.unsqueeze(dim=0)
    dec_pad = torch.zeros([enc_in.shape[0], pred_len, enc_in.shape[-1]]).float().to(device)
    dec_in = torch.cat((enc_in[:, -label_len:, :], dec_pad), dim=1)
    dec_in = dec_in.float().to(device)

    model.eval()
    with torch.no_grad():
        predict = model(enc_in, enc_mask, dec_in, dec_mask)
    predict = predict * 2.55054430 + 3.60226484
    predict = predict.cpu().numpy()
    return predict, hour

