import torch
import model.Informer as net
import pandas as pd
import utils.timefeature as timef

def eval(pred_data, checkpoint_path):
    seq_len = 720
    pred_len = 24
    label_len = 168
    input_size = 2
    output_size = 1
    mean = 3.57612728
    std = 2.64502802
    freq = '10min'
    device = torch.device('cuda:0')
    model = net.Informer(input_size, input_size, out_len = pred_len, c_out=output_size, freq='t')
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])  
    
    enc_in = (pred_data.iloc[:, 1:3].values -mean) /std
    pred_data['date'] = pd.to_datetime(pred_data.date)
    enc_mask = timef.time_features(pred_data, 0, 't')    
    enc_in = torch.from_numpy(enc_in).float().to(device)
    enc_in = enc_in.unsqueeze(dim=0)
    enc_mask = torch.from_numpy(enc_mask).float().to(device).unsqueeze(dim=0)
    dec_pad = torch.zeros([enc_in.shape[0], pred_len, enc_in.shape[-1]]).float().to(device)
    dec_in = torch.cat((enc_in[:, -label_len:, :], dec_pad), dim=1)
    dec_in = dec_in.float().to(device)
    pred_date = pd.date_range(pred_data.date.values[-1], periods=pred_len+1, freq=freq)
    hour = pred_date[-1]
    df_stamp = pd.DataFrame(columns = ['date'])
    df_stamp.date = list(pred_data.date.values[-label_len:]) + list(pred_date[1:])
    dec_mask = timef.time_features(df_stamp, 0, 't')
    dec_mask = torch.from_numpy(dec_mask).float().to(device).unsqueeze(dim=0)
    model.eval()
    with torch.no_grad():
        print(enc_in.shape, enc_mask.shape, dec_in.shape, dec_mask.shape)
        predict = model(enc_in, enc_mask, dec_in, dec_mask)
    predict = predict * std + mean
    predict = predict.cpu().numpy()
    return predict, hour
