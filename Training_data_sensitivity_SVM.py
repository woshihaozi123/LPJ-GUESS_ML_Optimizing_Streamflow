import numpy as np
from sklearn import svm
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

    datastart=2000.0
    dataend = 2015.0
    test_startyear=[2000,2003,2007,2010,2013]
    test_endyear =[2002,2005,2009,2012,2015]


    # choose dataset for training and testing
    #train_data_ratio =13.0/16.0  #0.7500 #0.8333  #0.6667  # Choose 80% of the data for testing
    for peroidid in range(0,5):
        test_data_ratio_start = float(test_startyear[peroidid] - datastart) / (dataend - datastart + 1)
        test_data_ratio_end = float(test_endyear[peroidid] - datastart + 1) / (dataend - datastart + 1)
        test_data_start = int(data_len * test_data_ratio_start)
        test_data_end = int(data_len * test_data_ratio_end)

        print([test_data_start, test_data_end])


        if peroidid==0:


            train_x = dataset[test_data_end+1:, 0:6]
            train_y = dataset[test_data_end+1:, 6]
            train_y_r = dataset[test_data_end+1:, 7]
            t_for_training = t[test_data_end+1:]
            # test_x = train_x
            # test_y = train_y
            test_x = dataset[:test_data_end+1, 0:6]
            test_y = dataset[:test_data_end+1, 6]
            test_y_r = dataset[:test_data_end+1, 7]
            t_for_testing = t[:test_data_end+1]

            LPJGUESSrunoff_train=LPJGUESSrunoff[test_data_end+1:]
            LPJGUESSrunoff_test=LPJGUESSrunoff[:test_data_end+1]

        elif peroidid==4:



            train_x = dataset[:test_data_start, 0:6]
            train_y = dataset[:test_data_start, 6]
            train_y_r = dataset[:test_data_start, 7]
            t_for_training = t[:test_data_start]
            # test_x = train_x
            # test_y = train_y
            test_x = dataset[test_data_start:, 0:6]
            test_y = dataset[test_data_start:, 6]
            test_y_r = dataset[test_data_start:, 7]
            t_for_testing = t[test_data_start:]

            LPJGUESSrunoff_train = LPJGUESSrunoff[:test_data_start]
            LPJGUESSrunoff_test = LPJGUESSrunoff[test_data_start:]

        else:


            train_x1 = dataset[:test_data_start, 0:6]
            train_x2 = dataset[test_data_end+1:, 0:6]
            train_x=np.concatenate((train_x1,train_x2), axis=0)

            train_y1 = dataset[:test_data_start, 6]
            train_y2 = dataset[test_data_end+1:, 6]
            train_y=np.concatenate((train_y1,train_y2), axis=0)

            train_y_r1 = dataset[:test_data_start, 7]
            train_y_r2 = dataset[test_data_end+1:, 7]
            train_y_r=np.concatenate((train_y_r1,train_y_r2), axis=0)

            t_for_training1 = t[:test_data_start]
            t_for_training2 = t[test_data_end+1:]
            t_for_training=np.concatenate((t_for_training1, t_for_training2), axis=0)


            test_x = dataset[test_data_start:test_data_end+1, 0:6]
            test_y = dataset[test_data_start:test_data_end+1, 6]
            test_y_r = dataset[test_data_start:test_data_end+1, 7]
            t_for_testing = t[test_data_start:test_data_end+1]

            LPJGUESSrunoff_train1 = LPJGUESSrunoff[:test_data_start]
            LPJGUESSrunoff_train2 = LPJGUESSrunoff[test_data_end+1:]
            LPJGUESSrunoff_train = np.concatenate((LPJGUESSrunoff_train1 ,LPJGUESSrunoff_train2), axis=0)

            LPJGUESSrunoff_test = LPJGUESSrunoff[test_data_start:test_data_end+1]

        INPUT_FEATURES_NUM = 6
        OUTPUT_FEATURES_NUM = 1


        # ----------------- train -------------------
        train_x_tensor = train_x # set batch size to 5
        train_y_r_tensor = train_y_r # set batch size to 5
        # prediction on test dataset
        test_x_tensor = test_x # set batch size to 5, the same value with the training set

        rf = svm.SVR()
        rf.fit(train_x_tensor , train_y_r_tensor)
        predictive_y_r_for_training = rf.predict(train_x_tensor)
        predictive_y_r_for_testing = rf.predict(test_x_tensor)

        # ----------------- save data -------------------
        prediction_trn = predictive_y_r_for_training + LPJGUESSrunoff_train
        prediction_tst = predictive_y_r_for_testing + LPJGUESSrunoff_test
        # Combine the lists into four columns
        combined_data_trn = list(zip(LPJGUESSrunoff_train, train_y, train_y_r, prediction_trn))
        combined_data_tst = list(zip(LPJGUESSrunoff_test, test_y, test_y_r, prediction_tst))
        columns_trn = ['LPJGUESS_trn', 'GRDC_trn', 'Residual_trn', 'Prediction_trn']
        columns_tst = ['LPJGUESS_tst', 'GRDC_tst', 'Residual_tst', 'Prediction_tst']
        MLR.write_to_excel_file( columns_trn, combined_data_trn, '.\output\Training_data_sensitivity\SVM_rc_Training'
                                 +str(test_startyear[peroidid])+'_'+str(test_endyear[peroidid])+'.xlsx', sheet_name = 'Sheet1')
        MLR.write_to_excel_file(columns_tst, combined_data_tst, '.\output\Training_data_sensitivity\SVM_rc_Test'
                                +str(test_startyear[peroidid])+'_'+str(test_endyear[peroidid])+'.xlsx', sheet_name='Sheet1')
        index=MLR.calIndex(test_y, prediction_tst.ravel())
        combined_data_index = list(zip([ 'NSE','R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], index))
        MLR.write_to_excel_file(['index','value'],combined_data_index , '.\output\Training_data_sensitivity\SVM_rc_Index'
                                +str(test_startyear[peroidid])+'_'+str(test_endyear[peroidid])+'.xlsx', sheet_name='Sheet1')
        print('index:', index)
        # ----------------- save data monthly index -------------------
        test_data_len = test_y.shape[0]

        daynums = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        numofyear = int(test_y.shape[0] / 365);

        # 按照每个月的天数切分数据并存入数组
        monthly_Obsrunoff = []
        monthly_Prerunoff = []
        monthly_Modelrunoff = []
        start_index = 0
        for yearid in np.linspace(1, numofyear, numofyear):
            for num_days in daynums:
                end_index = start_index + num_days
                monthly_Obsrunoff.append(sum(test_y[start_index:end_index]))
                monthly_Prerunoff.append(sum(prediction_tst[start_index:end_index]))
                monthly_Modelrunoff.append(sum(LPJGUESSrunoff_test[start_index:end_index]))
                start_index = end_index

        monthindex = MLR.calIndex(monthly_Obsrunoff, monthly_Prerunoff)
        combined_data_index = list(zip(['NSE', 'R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], monthindex))
        MLR.write_to_excel_file(['index', 'value'], combined_data_index, '.\output\Training_data_sensitivity\SVM_Index_monthly'
                                +str(test_startyear[peroidid])+'_'+str(test_endyear[peroidid])+'.xlsx', sheet_name='Sheet1')
        print('(ObsRunoff, PreRunoff)monthindex:', monthindex)
    # ----------------- plot -------------------
        # 设置图片的宽度和高度（单位为英寸）
        width_inch = 40.0  # 设置图片的宽度为10英寸
        height_inch = 6.0  # 设置图片的高度为6英寸

        # 创建一个新的图像，并设置大小
        plt.figure(figsize=(width_inch, height_inch))
        plt.plot(t_for_training, LPJGUESSrunoff_train, 'g', label='LPJGUESS_trn',linewidth=1.5)
        plt.plot(t_for_training, train_y, 'b', label='GRDC_trn',linewidth=1.5)
        plt.plot(t_for_training, train_y_r, 'r', label='Residual_trn', linewidth=1.5)
        plt.plot(t_for_training, prediction_trn, 'y--', label='Prediction_trn',linewidth=1.5)


        plt.plot(t_for_testing, LPJGUESSrunoff_test, 'c', label='LPJGUESS_tst',linewidth=1.5)
        plt.plot(t_for_testing, test_y, 'k', label='GRDC_tst',linewidth=1.5)
        plt.plot(t_for_testing, test_y_r, 'r', label='Residual_tst', linewidth=1.5)
        plt.plot(t_for_testing, prediction_tst, 'm--', label='Prediction_tst',linewidth=1.5)

        plt.xlabel('Day')
        plt.ylabel('Runoff')
        plt.xlim(t[0], t[-1])
        plt.ylim(-9, 9)
        plt.legend(loc='lower left')
        plt.text(10, 100, "train", size=15, alpha=1.0)
        plt.text(data_len, 100, "test", size=15, alpha=1.0)
        # 保存图像到文件
        plt.savefig('.\output\Training_data_sensitivity\SVM_residual_cliamte'
                    +str(test_startyear[peroidid])+'_'+str(test_endyear[peroidid])+'.png', dpi=600)
        plt.show()
