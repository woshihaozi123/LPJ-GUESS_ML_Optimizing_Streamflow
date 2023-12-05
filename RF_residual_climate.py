import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import MLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

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
    train_x_tensor = train_x # set batch size to 5
    train_y_r_tensor = train_y_r # set batch size to 5
    # prediction on test dataset
    test_x_tensor = test_x # set batch size to 5, the same value with the training set

    # 定义超参数范围
    rf_params = {
        'n_estimators': sp_randint(10, 100),
        'max_features': sp_randint(1, 64),
        'max_depth': sp_randint(5, 50),
        'min_samples_split': sp_randint(2, 11),
        'min_samples_leaf': sp_randint(1, 11)
        #,'criterion': ['squared_error', 'absolute_error']  # Adjusted for regression
    }
    # Create a Random Forest Regressor object
    clf_rf = RandomForestRegressor(random_state=0,n_jobs=-1)

    # Optimize hyperparameters using Randomized Search
    Random_search = RandomizedSearchCV(clf_rf, param_distributions=rf_params, n_iter=20, cv=5,
                                   scoring='neg_mean_squared_error')
    Random_search.fit(train_x_tensor, train_y_r_tensor)

    # save best_params into Excel file
    best_params = Random_search.best_params_
    print("best_params:", best_params)

    # convert best_params to DataFrame
    df_best_params = pd.DataFrame([best_params])

    # save into Excel file
    excel_filename = '.\output\RF_best_params.xlsx'
    df_best_params.to_excel(excel_filename, index=False)

    # Make predictions
    predictive_y_r_for_training = Random_search.predict(train_x_tensor)
    predictive_y_r_for_testing = Random_search.predict(test_x_tensor)

    #rf= RandomForestRegressor(n_estimators=1000,oob_score=True)
    #rf.fit(train_x_tensor , train_y_r_tensor)
    #predictive_y_r_for_training = rf.predict(train_x_tensor)
    #predictive_y_r_for_testing = rf.predict(test_x_tensor)

    # ----------------- save data -------------------
    prediction_trn = predictive_y_r_for_training + LPJGUESSrunoff[:train_data_len]
    prediction_tst = predictive_y_r_for_testing + LPJGUESSrunoff[train_data_len:]
    # Combine the lists into four columns
    combined_data_trn = list(zip(LPJGUESSrunoff[:train_data_len], train_y, train_y_r, prediction_trn))
    combined_data_tst = list(zip(LPJGUESSrunoff[train_data_len:], test_y, test_y_r, prediction_tst))
    columns_trn = ['LPJGUESS_trn', 'GRDC_trn', 'Residual_trn', 'Prediction_trn']
    columns_tst = ['LPJGUESS_tst', 'GRDC_tst', 'Residual_tst', 'Prediction_tst']
    MLR.write_to_excel_file( columns_trn, combined_data_trn, '.\output\RF_rc_Training.xlsx', sheet_name = 'Sheet1')
    MLR.write_to_excel_file(columns_tst, combined_data_tst, '.\output\RF_rc_Test.xlsx', sheet_name='Sheet1')
    index=MLR.calIndex(test_y, prediction_tst.ravel())
    combined_data_index = list(zip([ 'NSE','R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], index))
    MLR.write_to_excel_file(['index','value'],combined_data_index , '.\output\RF_rc_Index.xlsx', sheet_name='Sheet1')
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
    plt.savefig('.\output\RF_residual_cliamte.png', dpi=600)
    plt.show()

    # record end time
    end_time = time.time()
    # count and print time
    elapsed_time = end_time - start_time
    print(f"Runing time: {elapsed_time} second")
