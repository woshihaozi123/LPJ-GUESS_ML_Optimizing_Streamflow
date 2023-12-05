# -*- coding:UTF-8 -*-
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import MLR
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

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
    train_x = dataset[:train_data_len, 1:12]
    train_y = dataset[:train_data_len, 12]
    train_y_r = dataset[:train_data_len, 13]
    t_for_training = t[:train_data_len]

    # testing data
    test_x = dataset[train_data_len:, 1:12]
    test_y = dataset[train_data_len:, 12]
    test_y_r = dataset[train_data_len:, 13]
    t_for_testing = t[train_data_len:]

    # -----------------INPUT and output FEATURES number-----------------
    INPUT_FEATURES_NUM = 11
    OUTPUT_FEATURES_NUM = 1
    # ----------------- train -------------------
    #train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batch size to 5
    #train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 5

    # transfer data to pytorch tensor
    #train_x_tensor = torch.from_numpy(train_x_tensor)
    #train_y_tensor = torch.from_numpy(train_y_tensor)
    # test_x_tensor = torch.from_numpy(test_x)

    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    # 将 train_y 转换为二维数组
    train_y = train_y.reshape(-1, 1)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32)

    # prediction on test dataset
    #test_x_tensor = test_x.reshape(-1, 1,INPUT_FEATURES_NUM)  # set batch size to 5, the same value with the training set
    #test_x_tensor = torch.from_numpy(test_x_tensor)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    # 将 test_y 转换为二维数组
    test_y = test_y.reshape(-1, 1)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32)




    param_dist = {
        'hidden_size': sp_randint(10, 50),# 隐藏层单元数
        'num_layers': sp_randint(2, 5),# LSTM层的数量
        'lr': uniform(0.001, 0.1),# 学习率
        #'epochs': 100 #sp_randint(5, 50)  # 假设我们探索5到50之间的epochs值
    }
    max_epochs = 200

    n_iter_search = 20  # 进行多少次迭代
    best_score = np.inf
    best_params = None

    for i in range(n_iter_search):

        # 从分布中随机抽取参数
        params = {k: v.rvs() for k, v in param_dist.items()}

        # 创建并训练模型
        lstm_model = LstmRNN(INPUT_FEATURES_NUM, params['hidden_size'], OUTPUT_FEATURES_NUM, params['num_layers'])
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=params['lr'])

        # 训练模型
        loss_function = nn.MSELoss()

        for epoch in range(max_epochs):
            #print(epoch)
            lstm_model.train()  # 设置模型为训练模式
            output = lstm_model(train_x_tensor)# 将 train_x_tensor 转换为 (batch_size, seq_len, input_size)
            output = output.view(train_y_tensor.shape)

            # 计算损失
            loss = loss_function(output.squeeze(), train_y_tensor.squeeze()) # 将 output 转换为 (batch_size, output_size)
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if loss.item() < 0.05:

                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
                print("The loss value is reached")
                break
            elif (epoch + 1) % 100 == 0:
                print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            # 模型评估
        lstm_model.eval()  # 设置模型为评估模式
        with torch.no_grad():
             # 假设 val_x_tensor 和 val_y_tensor 是验证集数据
            val_output = lstm_model(test_x_tensor)  # 前向传播
            val_output = val_output.view(test_y_tensor.shape)
            val_loss = loss_function(val_output.squeeze(), test_y_tensor.squeeze())

        current_score = val_loss.item()

        # 保存最佳分数和参数
        if current_score < best_score:
            best_score = current_score
            best_params = params

    print("Best Score: ", best_score)
    print("Best Parameters: ", best_params)
    # convert best_params to DataFrame
    df_best_params = pd.DataFrame([best_params])
    # save into Excel file
    excel_filename = '.\output\LSTM_best_params.xlsx'
    df_best_params.to_excel(excel_filename, index=False)

    # Retrain the model using optimal parameters
    optimal_lstm_model = LstmRNN(INPUT_FEATURES_NUM, best_params['hidden_size'], OUTPUT_FEATURES_NUM,
                                 best_params['num_layers'])
    optimizer = torch.optim.Adam(optimal_lstm_model.parameters(), lr=best_params['lr'])

    # Retraining model
    for epoch in range(max_epochs):
        optimal_lstm_model.train()
        output = optimal_lstm_model(train_x_tensor)
        output = output.view(train_y_tensor.shape)
        loss = loss_function(output.squeeze(), train_y_tensor.squeeze())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if loss.item() < 0.05:
            break

    # Test with the best trained model
    optimal_lstm_model.eval()
    with torch.no_grad():
        predictive_y_for_training = optimal_lstm_model(train_x_tensor)
        predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
        predictive_y_for_testing = optimal_lstm_model(test_x_tensor)
        predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()


    # ----------------- save data -------------------
    prediction_trn=predictive_y_for_training
    prediction_tst=predictive_y_for_testing
    # Combine the lists into four columns
    combined_data_trn = list(zip(LPJGUESSrunoff[:train_data_len], train_y.ravel(), train_y.ravel(), prediction_trn.ravel()))
    combined_data_tst = list(zip(LPJGUESSrunoff[train_data_len:], test_y.ravel(), test_y.ravel(), prediction_tst.ravel()))

    columns_trn = ['LPJGUESS_trn', 'GRDC_trn', 'GRDC_trn', 'Prediction_trn']
    columns_tst = ['LPJGUESS_tst', 'GRDC_tst', 'GRDC_tst', 'Prediction_tst']

    MLR.write_to_excel_file(columns_trn, combined_data_trn, '.\output\LSTM_c_Train.xlsx', sheet_name='Sheet1')
    MLR.write_to_excel_file(columns_tst, combined_data_tst, '.\output\LSTM_c_Test.xlsx', sheet_name='Sheet1')

    index = MLR.calIndex(test_y.ravel(), prediction_tst.ravel())
    combined_data_index = list(zip([ 'NSE','R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], index))
    MLR.write_to_excel_file(['Index', 'Value'], combined_data_index, '.\output\LSTM_Index.xlsx', sheet_name='Sheet1')
    print('index [NSE,R, PBIAS, RMSE, RMSEper, MAE, MAEper, PAE]:')
    print(index)

    # ----------------- plot -------------------
    # 设置图片的宽度和高度（单位为英寸）
    width_inch = 40.0  # 设置图片的宽度为10英寸
    height_inch = 6.0  # 设置图片的高度为6英寸

    # 创建一个新的图像，并设置大小
    plt.figure(figsize=(width_inch, height_inch))
    plt.plot(t_for_training, LPJGUESSrunoff[:train_data_len].ravel(), 'g', label='LPJGUESS_trn',linewidth=1.5)
    plt.plot(t_for_training, train_y.ravel(), 'b', label='GRDC_trn',linewidth=1.5)

    plt.plot(t_for_training, prediction_trn.ravel(), 'y--', label='Prediction_trn',linewidth=1.5)


    plt.plot(t_for_testing, LPJGUESSrunoff[train_data_len:].ravel(), 'c', label='LPJGUESS_tst',linewidth=1.5)
    plt.plot(t_for_testing, test_y.ravel(), 'k', label='GRDC_tst',linewidth=1.5)

    plt.plot(t_for_testing, prediction_tst.ravel(), 'm--', label='Prediction_tst',linewidth=1.5)

    plt.plot([t[train_data_len], t[train_data_len]], [-6, 6], 'r--', label='separation line',linewidth=1.5)  # separation line

    plt.xlabel('Day')
    plt.ylabel('Runoff')
    plt.xlim(t[0], t[-1])
    plt.ylim(-9, 9)
    plt.legend(loc='upper right')
    plt.text(10, 100, "train", size=15, alpha=1.0)
    plt.text(data_len, 100, "test", size=15, alpha=1.0)
    # 保存图像到文件
    plt.savefig('.\output\LSTM_climate.png', dpi=600)
    plt.show()

    # record end time
    end_time = time.time()
    # count and print time
    elapsed_time = end_time - start_time
    print(f"Runing time: {elapsed_time} second")