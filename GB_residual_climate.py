import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import matplotlib.pyplot as plt
import MLR
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # create database
    # 读取 Excel 文件
    df = pd.read_excel('.\data\RunoffDailySeries20002015.xlsx')

    # 提取两列数据到数组
    LPJGUESSrunoff = df['LPJRunoff'].values
    GRDCrunoff = df['GRDCRunoff'].values
    Residualrunoff = df['Residual'].values

    df = pd.read_excel('.\data\ClimateDailySeries20002015.xlsx')
    Temp = df['Temp'].values
    Prec = df['Prec'].values
    Rad = df['Rad'].values
    U10 = df['U10'].values
    Relhum = df['Relhum'].values

    # 创建 Min-Max 归一化器
    scaler = MinMaxScaler()

    # 对每个特征进行 Min-Max 归一化
    Temp_normalized = scaler.fit_transform(Temp.reshape(-1, 1))
    Prec_normalized = scaler.fit_transform(Prec.reshape(-1, 1))
    Rad_normalized = scaler.fit_transform(Rad.reshape(-1, 1))
    U10_normalized = scaler.fit_transform(U10.reshape(-1, 1))
    Relhum_normalized = scaler.fit_transform(Relhum.reshape(-1, 1))
    LPJGUESSrunoff_normalized = scaler.fit_transform(LPJGUESSrunoff.reshape(-1, 1))

    data_len = LPJGUESSrunoff.shape[0]
    t = np.linspace(1, data_len, data_len)

    dataset = np.zeros((data_len, 8))
    dataset[:, 0] = LPJGUESSrunoff_normalized.reshape(-1, )
    dataset[:, 1] = Temp_normalized.reshape(-1, )
    dataset[:, 2] = Prec_normalized.reshape(-1, )
    dataset[:, 3] = Rad_normalized.reshape(-1, )
    dataset[:, 4] = U10_normalized.reshape(-1, )
    dataset[:, 5] = Relhum_normalized.reshape(-1, )

    dataset[:, 6] = GRDCrunoff
    dataset[:, 7] = Residualrunoff
    dataset = dataset.astype('float32')

    # plot part of the original dataset
    width_inch = 30.0  # 设置图片的宽度为10英寸
    height_inch = 6.0  # 设置图片的高度为6英寸

    # choose dataset for training and testing
    train_data_ratio =13.0/16.0  #0.7500 #0.8333  #0.6667  # Choose 80% of the data for testing
    train_data_len = int(data_len * train_data_ratio)
    train_x = dataset[:train_data_len, 0:6]
    train_y = dataset[:train_data_len, 6]
    train_y_r = dataset[:train_data_len, 7]
    t_for_training = t[:train_data_len]
    # test_x = train_x
    # test_y = train_y
    test_x = dataset[train_data_len:, 0:6]
    test_y = dataset[train_data_len:, 6]
    test_y_r = dataset[train_data_len:, 7]
    t_for_testing = t[train_data_len:]


    INPUT_FEATURES_NUM = 6
    OUTPUT_FEATURES_NUM = 1


    # ----------------- train -------------------
    train_x_tensor = train_x # set batch size to 5
    train_y_r_tensor = train_y_r # set batch size to 5
    # prediction on test dataset
    test_x_tensor = test_x # set batch size to 5, the same value with the training set

    rf= GradientBoostingRegressor(loss='quantile', alpha=0.3, max_depth=6)
    rf.fit(train_x_tensor , train_y_r_tensor)
    predictive_y_r_for_training = rf.predict(train_x_tensor)
    predictive_y_r_for_testing = rf.predict(test_x_tensor)

    # ----------------- save data -------------------
    prediction_trn = predictive_y_r_for_training + LPJGUESSrunoff[:train_data_len]
    prediction_tst = predictive_y_r_for_testing + LPJGUESSrunoff[train_data_len:]
    # Combine the lists into four columns
    combined_data_trn = list(zip(LPJGUESSrunoff[:train_data_len], train_y, train_y_r, prediction_trn))
    combined_data_tst = list(zip(LPJGUESSrunoff[train_data_len:], test_y, test_y_r, prediction_tst))
    columns_trn = ['LPJGUESS_trn', 'GRDC_trn', 'Residual_trn', 'Prediction_trn']
    columns_tst = ['LPJGUESS_tst', 'GRDC_tst', 'Residual_tst', 'Prediction_tst']
    MLR.write_to_excel_file( columns_trn, combined_data_trn, '.\output\GB_rc_Training.xlsx', sheet_name = 'Sheet1')
    MLR.write_to_excel_file(columns_tst, combined_data_tst, '.\output\GB_rc_Test.xlsx', sheet_name='Sheet1')
    index=MLR.calIndex(test_y, prediction_tst.ravel())
    combined_data_index = list(zip([ 'NSE','R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], index))
    MLR.write_to_excel_file(['index','value'],combined_data_index , '.\output\GB_rc_Index.xlsx', sheet_name='Sheet1')
    print('index:', index)
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
    plt.savefig('.\output\GB_residual_cliamte.png', dpi=600)
    plt.show()
