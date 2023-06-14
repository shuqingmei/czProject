import datetime
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import data.dataloader as dl
import model.Informer as net
import os 
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

TENSORBOARD = False  # 是否开启tensorboard曲线
if TENSORBOARD:
    writer1 = SummaryWriter(f'../log/board/train/', comment='Linear')  # trainloss曲线路径  
    writer2 = SummaryWriter(f'../log/board/test/', comment='Linear')   # testloss曲线路径

checkpoint_name = 'Informer'
seq_len = 360
pred_len = 24  # 预测长度
label_len = 180
input_size = 8  # 输入特征数量
output_size = 1

train_data = dl.DatasetLoader(seq_len, label_len, pred_len)
train_iter = Data.DataLoader(dataset=train_data, batch_size=100, shuffle=False, num_workers=0)
test_data = dl.DatasetLoader(seq_len, label_len, pred_len, flag=1)
test_iter = Data.DataLoader(dataset=test_data, batch_size=20, shuffle=False, num_workers=0)

device = torch.device('cuda:3')
model = net.Informer(input_size, input_size, out_len = pred_len, c_out=output_size, freq='t', dropout=0.1)
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(m.weight)

loss = nn.MSELoss()

lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.to(device)
epochs = 1000

train_loss_list = []
test_loss_list = []

for epoch in range(epochs):
    time1 = datetime.datetime.now()
    train_loss = 0.0
    batch_num = 0
    for enc_in, enc_mask, dec_in, dec_mask in train_iter:
        enc_in = enc_in.to(device).float()  # b step feature
        enc_mask = enc_mask.to(device).float()  #
        dec_in = dec_in.to(device).float()  #
        dec_mask = dec_mask.to(device).float()  #
        label = dec_in[:, label_len:, 1:2]
        dec_inp = torch.zeros([dec_in.shape[0], pred_len, dec_in.shape[-1]]).float().to(device)
        dec_in = torch.cat([dec_in[:,:label_len,:], dec_inp], dim=1).float()

        optimizer.zero_grad()
        pred = model(enc_in, enc_mask, dec_in, dec_mask)
        l = loss(pred, label)
        l.backward()
        optimizer.step()
        train_loss += l.item()
        batch_num += 1

    train_loss = train_loss/batch_num
    time3 = datetime.datetime.now()
    print(f'epoch{epoch} finished, loss = {train_loss}, time cost = {time3 - time1}')
    if TENSORBOARD:
        writer1.add_scalar('Informer', train_loss, epoch)
    if (epoch+1) % 1 == 0:
        model.eval()
        test_loss = 0
        batch_num = 0
        for enc_in, enc_mask, dec_in, dec_mask in test_iter:
            with torch.no_grad():
                enc_in = enc_in.to(device).float()  # b step feature
                enc_mask = enc_mask.to(device).float()  #
                dec_in = dec_in.to(device).float()  #
                dec_mask = dec_mask.to(device).float()  #
                label = dec_in[:, label_len:, 1:2]
                dec_inp = torch.zeros([dec_in.shape[0], pred_len, dec_in.shape[-1]]).float().to(device)
                dec_in = torch.cat([dec_in[:,:label_len,:], dec_inp], dim=1).float()
                pred = model(enc_in, enc_mask, dec_in, dec_mask)
                # label = test_data.inverse_transform(label)
                # pred = test_data.inverse_transform(pred)

                l = loss(pred, label)
                test_loss += l.item()
                batch_num += 1
        test_loss = test_loss/batch_num
        print('test_loss == ', test_loss)
        if TENSORBOARD:
            writer1.add_scalar('Informer', test_loss, epoch)

        model.train()
        # 保存checkpoint
        # checkpoint = {
        #     'epoch': epoch,
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        # if not os.path.exists(f'../log/checkpoint/{checkpoint_name}'):
        #     os.makedirs(f'../log/checkpoint/{checkpoint_name}')
        # torch.save(checkpoint, f'../log/checkpoint/{checkpoint_name}/{epoch+1}.pth')
