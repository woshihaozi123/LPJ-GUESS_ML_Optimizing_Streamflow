# -*- coding:UTF-8 -*-
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import MLR
from sklearn.preprocessing import MinMaxScaler

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
        super(LstmRNN,self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x 的形状为 (batch_size, seq_len, input_size)
        # LSTM 输入的形状为 (batch_size, seq_len, input_size)
        lstm_output, _ = self.lstm(x)
        # lstm_output 的形状为 (batch_size, seq_len, hidden_size)
        output = self.forwardCalculation(lstm_output)  # 取最后一个时间步的输出进行预测
        # output 的形状为 (batch_size, output_size)
        return output


if __name__ == '__main__':
    import time

    # start to record time
    start_time = time.time()

    ### -----------------read Excel files-----------------
    # get colmn data by name
    df = pd.read_excel('.\data\RunoffDailySeries19662015.xlsx')  # Runoff variables
    LPJGUESSrunoff = df['LPJRunoff'].values
    GRDCrunoff = df['GRDCRunoff'].values
    Residualrunoff = df['Residual'].values

    df = pd.read_excel('.\data\ClimateDailySeries19662015.xlsx')  # Climate variables
    Temp = df['Temp'].values
    Prec = df['Prec'].values
    Rad = df['Rad'].values
    U10 = df['U10'].values
    Relhum = df['Relhum'].values
    MinTemp = df['MinTemp'].values
    MaxTemp = df['MaxTemp'].values
    CO2 = df['CO2'].values
    CroplandProp = df['CroplandProp'].values
    PastureProp = df['PastureProp'].values
    NaturalProp = df['NaturalProp'].values

    # -----------------Normalizer-----------------
    # create Min-Max Normalizer
    scaler = MinMaxScaler()

    # Normalize each X variables by Min-Max Normalizer
    Temp_normalized = scaler.fit_transform(Temp.reshape(-1, 1))
    Prec_normalized = scaler.fit_transform(Prec.reshape(-1, 1))
    Rad_normalized = scaler.fit_transform(Rad.reshape(-1, 1))
    U10_normalized = scaler.fit_transform(U10.reshape(-1, 1))
    Relhum_normalized = scaler.fit_transform(Relhum.reshape(-1, 1))
    MinTemp_normalized = scaler.fit_transform(MinTemp.reshape(-1, 1))
    MaxTemp_normalized = scaler.fit_transform(MaxTemp.reshape(-1, 1))
    CO2_normalized = scaler.fit_transform(CO2.reshape(-1, 1))
    CroplandProp_normalized = scaler.fit_transform(CroplandProp.reshape(-1, 1))
    PastureProp_normalized = scaler.fit_transform(PastureProp.reshape(-1, 1))
    NaturalProp_normalized = scaler.fit_transform(NaturalProp.reshape(-1, 1))

    LPJGUESSrunoff_normalized = scaler.fit_transform(LPJGUESSrunoff.reshape(-1, 1))

    ###----------------- add all variables into a datasets for uaage-----------------
    data_len = LPJGUESSrunoff.shape[0]
    t = np.linspace(1, data_len, data_len)
    dataset = np.zeros((data_len, 8 + 6))

    # X variables
    dataset[:, 0] = LPJGUESSrunoff_normalized.reshape(-1, )
    dataset[:, 1] = Temp_normalized.reshape(-1, )
    dataset[:, 2] = Prec_normalized.reshape(-1, )
    dataset[:, 3] = Rad_normalized.reshape(-1, )
    dataset[:, 4] = U10_normalized.reshape(-1, )
    dataset[:, 5] = Relhum_normalized.reshape(-1, )
    dataset[:, 6] = MinTemp_normalized.reshape(-1, )
    dataset[:, 7] = MaxTemp_normalized.reshape(-1, )
    dataset[:, 8] = CO2_normalized.reshape(-1, )
    dataset[:, 9] = CroplandProp_normalized.reshape(-1, )
    dataset[:, 10] = PastureProp_normalized.reshape(-1, )
    dataset[:, 11] = NaturalProp_normalized.reshape(-1, )

    # y variables
    dataset[:, 12] = GRDCrunoff
    dataset[:, 13] = Residualrunoff
    dataset = dataset.astype('float32')

    ###----------------- divide dataset for training and testing-----------------
    train_data_ratio = 40.0 / 50.0  # 0.7500 #0.8333  #0.6667  # Choose 80% of the data for testing
    train_data_len = int(data_len * train_data_ratio)

    # traing data
    train_x = dataset[:train_data_len, 0:12]
    train_y = dataset[:train_data_len, 12]
    train_y_r = dataset[:train_data_len, 13]
    t_for_training = t[:train_data_len]

    # testing data
    test_x = dataset[train_data_len:, 0:12]
    test_y = dataset[train_data_len:, 12]
    test_y_r = dataset[train_data_len:, 13]
    t_for_testing = t[train_data_len:]

    # -----------------INPUT and output FEATURES number-----------------
    INPUT_FEATURES_NUM = 12
    OUTPUT_FEATURES_NUM = 1
    # ----------------- train -------------------
    #train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batch size to 5
    #train_y_r_tensor = train_y_r.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 5

    # transfer data to pytorch tensor
    #train_x_tensor = torch.from_numpy(train_x_tensor)
    #train_y_r_tensor = torch.from_numpy(train_y_r_tensor)
    # test_x_tensor = torch.from_numpy(test_x)
    # 将 train_y 转换为二维数组
    train_y_r = train_y_r.reshape(-1, 1)
    # 将 test_y 转换为二维数组
    test_y_r = test_y_r.reshape(-1, 1)

    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_r_tensor = torch.tensor(train_y_r, dtype=torch.float32)

    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 47, output_size=OUTPUT_FEATURES_NUM, num_layers=3)  # 16 hidden units
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0713)

    max_epochs = 200
    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor)# 将 train_x_tensor 转换为 (batch_size, seq_len, input_size)
        output = output.view(train_y_r_tensor.shape)
        loss = loss_function(output.squeeze(), train_y_r_tensor.squeeze()) # 将 output 转换为 (batch_size, output_size)

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
    prediction_trn=predictive_y_r_for_training+LPJGUESSrunoff[:train_data_len].reshape(-1, 1)
    prediction_tst=predictive_y_r_for_testing + LPJGUESSrunoff[train_data_len:].reshape(-1, 1)
    # Combine the lists into four columns
    combined_data_trn = list(zip(LPJGUESSrunoff[:train_data_len], train_y, train_y_r.ravel(), prediction_trn.ravel()))
    combined_data_tst = list(zip(LPJGUESSrunoff[train_data_len:], test_y, test_y_r.ravel(), prediction_tst.ravel()))

    columns_trn = ['LPJGUESS_trn', 'GRDC_trn', 'Residual_trn', 'Prediction_trn']
    columns_tst = ['LPJGUESS_tst', 'GRDC_tst', 'Residual_tst', 'Prediction_tst']

    MLR.write_to_excel_file(columns_trn, combined_data_trn, '.\output\LSTM_Training_r.xlsx', sheet_name='Sheet1')
    MLR.write_to_excel_file(columns_tst, combined_data_tst, '.\output\LSTM_Test_r.xlsx', sheet_name='Sheet1')

    index = MLR.calIndex(test_y, prediction_tst.ravel())
    combined_data_index = list(zip([ 'NSE','R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], index))
    MLR.write_to_excel_file(['Index', 'Value'], combined_data_index, '.\output\LSTM_Index_r.xlsx', sheet_name='Sheet1')
    print('index [NSE,R, PBIAS, RMSE, RMSEper, MAE, MAEper, PAE]:')
    print(index)

    # ----------------- plot -------------------
    # 设置图片的宽度和高度（单位为英寸）
    width_inch = 40.0  # 设置图片的宽度为10英寸
    height_inch = 6.0  # 设置图片的高度为6英寸

    # 创建一个新的图像，并设置大小
    plt.figure(figsize=(width_inch, height_inch))
    plt.plot(t_for_training, LPJGUESSrunoff[:train_data_len], 'g', label='LPJGUESS_trn',linewidth=1.5)
    plt.plot(t_for_training, train_y, 'b', label='GRDC_trn',linewidth=1.5)
    plt.plot(t_for_training, train_y_r, 'r', label='Residual_trn', linewidth=1.5)
    plt.plot(t_for_training, prediction_trn, 'y--', label='Prediction_trn',linewidth=1.5)


    plt.plot(t_for_testing, LPJGUESSrunoff[train_data_len:], 'c', label='LPJGUESS_tst',linewidth=1.5)
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
    plt.savefig('.\output\LSTM_residual_climate.png', dpi=600)
    plt.show()