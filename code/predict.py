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
    # 将日期或时间戳的字符串表示转换为日期时间对象，从而更容易对与时间相关的数据进行操作和分析
    enc_mask = timef.time_features(pred_data, 0, 't')
    # 调用了名为time_features的函数，它的作用是基于输入数据pred_data创建一个时间特征，返回的enc_mask是一个numpy数组
    enc_mask = torch.from_numpy(enc_mask).float().to(device).unsqueeze(dim=0)
    # 创建一个编码器掩码
    pred_date = pd.date_range(pred_data.date.values[-1], periods=pred_len+1, freq=freq)
    # pred_data.date.values[-1]表示从pred_data数据中获取最后一个日期值，periods=pred_len+1指定了要生成的日期范围的长度，freq=freq表示日期范围的频率
    # pred_date将生成一个日期范围，从pred_data的最后一个日期值开始，持续pred_len+1个时间间隔，并且时间间隔由freq参数确定
    hour = pred_date[-1]
    df_stamp = pd.DataFrame(columns = ['date'])
    # 创建了一个名为df_stamp的空数据框（DataFrame），只包含一个名为'date'的列
    df_stamp.date = list(pred_data.date.values[-label_len:]) + list(pred_date[1:])
    # 将df_stamp的'date'列填充为预测数据（pred_data）的最后label_len个日期值加上预测日期（pred_date）的第一个小时之后的所有日期值
    dec_mask = timef.time_features(df_stamp, 0, 't')
    dec_mask = torch.from_numpy(dec_mask).float().to(device).unsqueeze(dim=0)
    # 根据时间戳数据生成解码器掩码（dec_mask）
    model = net.Informer(input_size, input_size, out_len = pred_len, c_out=output_size, freq='t')
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    # 加载checkpoint中保存的模型权重参数到model对象中，使得模型具有预训练时的状态

    df = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_max.csv')
    df_min = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_min.csv')
    df_mean = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_mean.csv')
    df_var = pd.read_csv('/mnt/nfsData10/GuoLongZhao/czProject/dataset/项目部超声波风速风向_10T_var.csv')
    df_max = df.iloc[:, 1:3].values
    df_min = df_min.iloc[:, 1:3].values
    df_var = df_var.iloc[:, 1:3].values
    df_mean = df_mean.iloc[:, 1:3].values
    # 选择所有行的第1列到第2列的数据
    pred_data = np.concatenate((df_max, df_min, df_mean, df_var), axis=1)[0:seq_len, :]
    # 四个NumPy数组按列进行拼接，并且选择前360行的数据，然后将结果保存在pred_data变量中
    mean = np.average(pred_data, axis=0)
    std = np.std(pred_data, axis=0)
    print(pred_data[:, 1])
    enc_in = (pred_data -mean) /std
    # 输入数据标准化
    enc_in = torch.from_numpy(enc_in).float().to(device)
    enc_in = enc_in.unsqueeze(dim=0)
    # 编码器输入数据
    dec_pad = torch.zeros([enc_in.shape[0], pred_len, enc_in.shape[-1]]).float().to(device)
    dec_in = torch.cat((enc_in[:, -label_len:, :], dec_pad), dim=1)
    # 解码器输入数据，会拼接一段编码器输入数据
    dec_in = dec_in.float().to(device)

    model.eval()
    with torch.no_grad():
        predict = model(enc_in, enc_mask, dec_in, dec_mask)
    predict = predict * 2.55054430 + 3.60226484
    # 反标准化
    predict = predict.cpu().numpy()
    return predict, hour

