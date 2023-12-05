# -*- coding:UTF-8 -*-
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import MLR

# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        dropout_prob = 0.2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.dropout = nn.Dropout(p=dropout_prob)
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.dropout(x)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x


if __name__ == '__main__':
    # create database
    # 读取 Excel 文件
    df = pd.read_excel('.\data\RunoffDailySeries.xlsx')

    # 提取两列数据到数组
    LPJGUESSrunoff = df['LPJRunoff'].values
    GRDCrunoff = df['GRDCRunoff'].values
    Residualrunoff=df['Residual'].values
    data_len = LPJGUESSrunoff.shape[0]
    t = np.linspace(1, data_len, data_len)


    dataset = np.zeros((data_len, 3))
    dataset[:, 0] = LPJGUESSrunoff
    dataset[:, 1] = GRDCrunoff
    dataset[:, 2] = Residualrunoff
    dataset = dataset.astype('float32')

    # plot part of the original dataset
    width_inch = 30.0  # 设置图片的宽度为10英寸
    height_inch = 6.0  # 设置图片的高度为6英寸

    # 创建一个新的图像，并设置大小
    plt.figure(figsize=(width_inch, height_inch))
    plt.plot(t, dataset[:, 0], label='LPJGUESSrunoff')
    plt.plot(t, dataset[:, 1], label='GRDCrunoff')
    #plt.plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = 2.5')  # t = 2.5
    #plt.plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t = 6.8')  # t = 6.8
    plt.xlabel('month')
    #plt.ylim(-1.2, 1.2)
    plt.ylabel('LPJGUESSrunoff and GRDCrunoff')
    plt.legend(loc='upper right')

    # choose dataset for training and testing
    train_data_ratio = 0.8333  #0.6667  # Choose 80% of the data for testing
    train_data_len = int(data_len * train_data_ratio)
    train_x = dataset[:train_data_len, 0]
    train_y = dataset[:train_data_len, 1]
    train_y_r = dataset[:train_data_len, 2]
    INPUT_FEATURES_NUM = 1
    OUTPUT_FEATURES_NUM = 1
    t_for_training = t[:train_data_len]

    # test_x = train_x
    # test_y = train_y
    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    test_y_r = dataset[train_data_len:, 2]
    t_for_testing = t[train_data_len:]

    # ----------------- train -------------------
    train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batch size to 5
    train_y_r_tensor = train_y_r.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 5

    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_r_tensor = torch.from_numpy(train_y_r_tensor)
    # test_x_tensor = torch.from_numpy(test_x)

    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 20, output_size=OUTPUT_FEATURES_NUM, num_layers=2)  # 16 hidden units
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

    max_epochs = 10000
    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor)
        loss = loss_function(output, train_y_r_tensor)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if loss.item() < 0.05:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    # prediction on training dataset
    predictive_y_r_for_training = lstm_model(train_x_tensor)
    predictive_y_r_for_training = predictive_y_r_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # torch.save(lstm_model.state_dict(), 'model_params.pkl') # save model parameters to files

    # ----------------- test -------------------
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # load model parameters from files
    lstm_model = lstm_model.eval()  # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 1,
                                   INPUT_FEATURES_NUM)  # set batch size to 5, the same value with the training set
    test_x_tensor = torch.from_numpy(test_x_tensor)

    predictive_y_r_for_testing = lstm_model(test_x_tensor)
    predictive_y_r_for_testing = predictive_y_r_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ----------------- save data -------------------
    prediction_trn=predictive_y_r_for_training+train_x.reshape(-1, 1)
    prediction_tst=predictive_y_r_for_testing + test_x.reshape(-1, 1)
    # Combine the lists into four columns
    combined_data_trn = list(zip(train_x, train_y, train_y_r.ravel(), prediction_trn.ravel()))
    combined_data_tst = list(zip(test_x, test_y, test_y_r.ravel(), prediction_tst.ravel()))
    columns_trn = ['LPJGUESS_trn', 'GRDC_trn', 'Residual_trn', 'Prediction_trn']
    columns_tst = ['LPJGUESS_tst', 'GRDC_tst', 'Residual_tst', 'Prediction_tst']
    MLR.write_to_excel_file(columns_trn, combined_data_trn, 'LSTM_Training.xlsx', sheet_name='Sheet1')
    MLR.write_to_excel_file(columns_tst, combined_data_tst, 'LSTM_Test.xlsx', sheet_name='Sheet1')
    index = MLR.calIndex(test_y, prediction_tst.ravel())
    combined_data_index = list(zip([ 'NSE','R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], index))
    MLR.write_to_excel_file(['Index', 'Value'], combined_data_index, 'LSTM_Index.xlsx', sheet_name='Sheet1')
    print('index:',index)

    # ----------------- plot -------------------
    # 设置图片的宽度和高度（单位为英寸）
    width_inch = 30.0  # 设置图片的宽度为10英寸
    height_inch = 6.0  # 设置图片的高度为6英寸

    # 创建一个新的图像，并设置大小
    plt.figure(figsize=(width_inch, height_inch))
    plt.plot(t_for_training, train_x, 'g', label='LPJGUESS_trn',linewidth=1.5)
    plt.plot(t_for_training, train_y, 'b', label='GRDC_trn',linewidth=1.5)
    plt.plot(t_for_training, train_y_r, 'r', label='Residual_trn', linewidth=1.5)
    plt.plot(t_for_training, prediction_trn, 'y--', label='Prediction_trn',linewidth=1.5)


    plt.plot(t_for_testing, test_x, 'c', label='LPJGUESS_tst',linewidth=1.5)
    plt.plot(t_for_testing, test_y, 'k', label='GRDC_tst',linewidth=1.5)
    plt.plot(t_for_testing, test_y_r, 'r', label='Residual_tst', linewidth=1.5)
    plt.plot(t_for_testing, prediction_tst, 'm--', label='Prediction_tst',linewidth=1.5)

    plt.plot([t[train_data_len], t[train_data_len]], [-6, 6], 'r--', label='separation line',linewidth=1.5)  # separation line

    plt.xlabel('Day')
    plt.ylabel('Runoff')
    plt.xlim(t[0], t[-1])
    plt.ylim(-9, 9)
    plt.legend(loc='upper right')
    plt.text(10, 100, "train", size=15, alpha=1.0)
    plt.text(data_len, 100, "test", size=15, alpha=1.0)
    # 保存图像到文件
    plt.savefig('LSTM_residual.png', dpi=600)
    plt.show()