import numpy as np
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import MLR
from sklearn.preprocessing import MinMaxScaler
import shap
import sklearn
from sklearn.ensemble import RandomForestRegressor
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
    train_data_ratio = 50.0 / 50.0  # 0.7500 #0.8333  #0.6667  # Choose 80% of the data for testing
    train_data_len = int(data_len * train_data_ratio)

    # traing data
    train_x = dataset[:train_data_len, 1:12]
    train_y = dataset[:train_data_len, 0]
    train_y_r = dataset[:train_data_len, 13]
    t_for_training = t[:train_data_len]

    # testing data
    test_x = dataset[train_data_len:, 1:12]
    test_y = dataset[train_data_len:, 0]
    test_y_r = dataset[train_data_len:, 13]
    t_for_testing = t[train_data_len:]


    # ----------------- train -------------------
    train_x_tensor = train_x # set batch size to 5
    train_y_r_tensor = train_y # set batch size to 5
    # prediction on test dataset
    test_x_tensor = test_x # set batch size to 5, the same value with the training set

    X=train_x#dataset[:, 1:6]
    y=train_y#dataset[:, 0]

    # 定义特征名称
    feature_names = ['Temp', 'Prec', 'Rad', 'U10', 'Relhum','MinTemp','MaxTemp','CO2','CroplandProp','PastureProp','NaturalProp']

    # 创建一个dataframe,列名就是特征名
    df_x = pd.DataFrame(X, columns=feature_names)

    # 将dataframe转换为array
    X_new = df_x.to_numpy()

    feature_names_y=['Runoff']
    # 创建一个dataframe,列名就是特征名
    df_y = pd.DataFrame(y, columns=feature_names_y)

    # 将dataframe转换为array
    y_new = df_y.to_numpy().ravel()

    # 定义超参数范围
    rf_params = {
        'n_estimators': sp_randint(10, 100),
        'max_features': sp_randint(1, 64),
        'max_depth': sp_randint(5, 50),
        'min_samples_split': sp_randint(2, 11),
        'min_samples_leaf': sp_randint(1, 11)
        # ,'criterion': ['squared_error', 'absolute_error']  # Adjusted for regression
    }
    # Create a Random Forest Regressor object
    clf_rf = RandomForestRegressor(random_state=0, n_jobs=-1)

    # Optimize hyperparameters using Randomized Search
    Random_search = RandomizedSearchCV(clf_rf, param_distributions=rf_params, n_iter=20, cv=5,
                                       scoring='neg_mean_squared_error')
    Random_search.fit(X_new, y_new)

    # save best_params into Excel file
    best_params = Random_search.best_params_
    print("best_params:", best_params)

    # convert best_params to DataFrame
    df_best_params = pd.DataFrame([best_params])

    # save into Excel file
    excel_filename = '.\output\RF_best_params_forSHAP.xlsx'
    df_best_params.to_excel(excel_filename, index=False)

    # Make predictions


    model = RandomForestRegressor(**best_params, random_state=0, n_jobs=-1)
    model_prediction = model.fit(X_new, y_new)

    # 将背景数据汇总为K个样本
    K = 5  # 自定义K的值，可以根据需求调整
    background_data = shap.sample(X_new, K)  # 或者使用 shap.kmeans(train_x_tensor, K)
    # 创建SHAP的KernelExplainer对象
    explainer = shap.KernelExplainer(model.predict, background_data)

    # 计算SHAP值
    shap_values = explainer.shap_values(X_new)
    print(shap_values.shape)

    dpi = 600  # 设置所需的 DPI 值
    # 绘制 SHAP 值的摘要图
    fig = plt.figure()
    shap.summary_plot(shap_values, X_new, feature_names=feature_names)
    #plt.title(f'SHAP summary Plot')
    fig.savefig(f'.\output\SHAP summary Plot.png', dpi=dpi)  # 保存为 PNG 文件
    plt.close(fig)

    fig= plt.figure()
    shap.summary_plot(shap_values, X_new, plot_type="bar", feature_names=feature_names)#
    #plt.title(f'SHAP summary bar Plot')
    fig.savefig(f'.\output\SHAP summary bar Plot.png', dpi=dpi)  # 保存为 PNG 文件
    plt.close(fig)
    #plt.show()  # 显示图形
    #plt.clf()


    # 画 SHAP 相关依赖图
    #['Temp', 'Prec', 'Rad', 'U10', 'Relhum','MinTemp','MaxTemp','CO2','CroplandProp','PastureProp','NaturalProp']
    for feature_name in feature_names:
        # 绘制 SHAP 依赖图
        fig,ax= plt.subplots(figsize=(10, 8))
        shap.dependence_plot(feature_name, shap_values, X_new, interaction_index=None, feature_names=feature_names, ax=ax)
        #plt.tight_layout()
        #plt.title(f'SHAP Dependence Plot for {feature_name}')
        #fig = plt.gcf()  # 获取当前图形
        fig.savefig(f'.\output\shap_dependence_{feature_name}.png', dpi=dpi)  # 保存为 PNG 文件
        plt.close(fig)


        # 如果需要，也可以绘制与其他特征的互作图
        # 这里以 Radiation (Rad) 为例
        if feature_name != 'Temp':  # 避免与自身交互
            fig,ax= plt.subplots(figsize=(10, 8))
            shap.dependence_plot(feature_name, shap_values, X_new, interaction_index='Temp', feature_names=feature_names, ax=ax)
            #plt.title(f'SHAP Interaction Plot between {feature_name} and Temp')
            #= plt.gcf()  # 获取当前图形
            fig.savefig(f'.\output\shap_interaction_{feature_name}_Temp.png', dpi=dpi)
            plt.close(fig)

        if feature_name != 'Prec':  # 避免与自身交互
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.dependence_plot(feature_name, shap_values, X_new, interaction_index='Prec', feature_names=feature_names, ax=ax)
            #plt.title(f'SHAP Interaction Plot between {feature_name} and Prec')
            #fig = plt.gcf()  # 获取当前图形
            fig.savefig(f'.\output\shap_interaction_{feature_name}_Prec.png', dpi=dpi)
            plt.close(fig)

        if feature_name != 'Rad':  # 避免与自身交互
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.dependence_plot(feature_name, shap_values, X_new, interaction_index='Rad', feature_names=feature_names, ax=ax)
            #plt.title(f'SHAP Interaction Plot between {feature_name} and Rad')
            #fig = plt.gcf()  # 获取当前图形
            fig.savefig(f'.\output\shap_interaction_{feature_name}_Rad.png', dpi=dpi)
            plt.close(fig)

        if feature_name != 'U10':  # 避免与自身交互
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.dependence_plot(feature_name, shap_values, X_new, interaction_index='U10', feature_names=feature_names, ax=ax)
            #plt.title(f'SHAP Interaction Plot between {feature_name} and U10')
            #fig = plt.gcf()  # 获取当前图形
            fig.savefig(f'.\output\shap_interaction_{feature_name}_U10.png', dpi=dpi)
            plt.close(fig)

        if feature_name != 'Relhum':  # 避免与自身交互
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.dependence_plot(feature_name, shap_values, X_new, interaction_index='Relhum', feature_names=feature_names, ax=ax)
            #plt.title(f'SHAP Interaction Plot between {feature_name} and Relhum')
            #fig = plt.gcf()  # 获取当前图形
            fig.savefig(f'.\output\shap_interaction_{feature_name}_Relhum.png', dpi=dpi)
            plt.close(fig)

            if feature_name != 'MinTemp':  # 避免与自身交互
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.dependence_plot(feature_name, shap_values, X_new, interaction_index='MinTemp',
                                     feature_names=feature_names, ax=ax)
                # plt.title(f'SHAP Interaction Plot between {feature_name} and Relhum')
                # fig = plt.gcf()  # 获取当前图形
                fig.savefig(f'.\output\shap_interaction_{feature_name}_MinTemp.png', dpi=dpi)
                plt.close(fig)

            if feature_name != 'MaxTemp':  # 避免与自身交互
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.dependence_plot(feature_name, shap_values, X_new, interaction_index='MaxTemp',
                                     feature_names=feature_names, ax=ax)
                # plt.title(f'SHAP Interaction Plot between {feature_name} and Relhum')
                # fig = plt.gcf()  # 获取当前图形
                fig.savefig(f'.\output\shap_interaction_{feature_name}_MaxTemp.png', dpi=dpi)
                plt.close(fig)

        if feature_name != 'CO2':  # 避免与自身交互
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.dependence_plot(feature_name, shap_values, X_new, interaction_index='CO2',
                                 feature_names=feature_names, ax=ax)
            # plt.title(f'SHAP Interaction Plot between {feature_name} and Relhum')
            # fig = plt.gcf()  # 获取当前图形
            fig.savefig(f'.\output\shap_interaction_{feature_name}_CO2.png', dpi=dpi)
            plt.close(fig)

        if feature_name != 'CroplandProp':  # 避免与自身交互
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.dependence_plot(feature_name, shap_values, X_new, interaction_index='CroplandProp',
                                 feature_names=feature_names, ax=ax)
            # plt.title(f'SHAP Interaction Plot between {feature_name} and Relhum')
            # fig = plt.gcf()  # 获取当前图形
            fig.savefig(f'.\output\shap_interaction_{feature_name}_CroplandProp.png', dpi=dpi)
            plt.close(fig)

        if feature_name != 'PastureProp':  # 避免与自身交互
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.dependence_plot(feature_name, shap_values, X_new, interaction_index='PastureProp',
                                 feature_names=feature_names, ax=ax)
            # plt.title(f'SHAP Interaction Plot between {feature_name} and Relhum')
            # fig = plt.gcf()  # 获取当前图形
            fig.savefig(f'.\output\shap_interaction_{feature_name}_PastureProp.png', dpi=dpi)
            plt.close(fig)

        if feature_name != 'NaturalProp':  # 避免与自身交互
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.dependence_plot(feature_name, shap_values, X_new, interaction_index='NaturalProp',
                                 feature_names=feature_names, ax=ax)
            # plt.title(f'SHAP Interaction Plot between {feature_name} and Relhum')
            # fig = plt.gcf()  # 获取当前图形
            fig.savefig(f'.\output\shap_interaction_{feature_name}_NaturalProp.png', dpi=dpi)
            plt.close(fig)






    #shap.dependence_plot(1, shap_values, test_x_tensor)
    #shap.dependence_plot(2, shap_values, test_x_tensor)
    #shap.dependence_plot(3, shap_values, test_x_tensor)
    #shap.dependence_plot(4, shap_values, test_x_tensor)
    # 添加图标标题
    #plt.show()



    #svc_linear = sklearn.svm.SVC(kernel='rbf', probability=True)
    #svc_linear.fit(train_x_tensor , train_y_r_tensor)
    #print_accuracy(svc_linear.predict)
    # explain all the predictions in the test set
    #explainer = shap.KernelExplainer(svc_linear.predict_proba, train_x_tensor)
    #shap_values = explainer.shap_values(test_x_tensor)


    #predictive_y_r_for_training = rf.predict(train_x_tensor)
    #predictive_y_r_for_testing = rf.predict(test_x_tensor)
    #explainer = shap.KernelExplainer(rf.predict, train_x_tensor )
    #shap_values = explainer.shap_values(test_x_tensor)
    #shap.summary_plot(shap_values, train_x_tensor, feature_names=['Temp','Prec','Rad','U10','Relhum'])
    #shap.summary_plot(shap_values, train_x_tensor, plot_type="bar",feature_names=['Temp','Prec','Rad','U10','Relhum'])

    #shap.force_plot(explainer.expected_value, shap_values[0, :], test_x_tensor[0, :], feature_names=['Temp','Prec','Rad','U10','Relhum'])






#     # ----------------- save data -------------------
#     prediction_trn = predictive_y_r_for_training + LPJGUESSrunoff[:train_data_len]
#     prediction_tst = predictive_y_r_for_testing + LPJGUESSrunoff[train_data_len:]
#     # Combine the lists into four columns
#     combined_data_trn = list(zip(LPJGUESSrunoff[:train_data_len], train_y, train_y_r, prediction_trn))
#     combined_data_tst = list(zip(LPJGUESSrunoff[train_data_len:], test_y, test_y_r, prediction_tst))
#     columns_trn = ['LPJGUESS_trn', 'GRDC_trn', 'Residual_trn', 'Prediction_trn']
#     columns_tst = ['LPJGUESS_tst', 'GRDC_tst', 'Residual_tst', 'Prediction_tst']
#     MLR.write_to_excel_file( columns_trn, combined_data_trn, '.\output\SVM_rc_Training.xlsx', sheet_name = 'Sheet1')
#     MLR.write_to_excel_file(columns_tst, combined_data_tst, '.\output\SVM_rc_Test.xlsx', sheet_name='Sheet1')
#     index=MLR.calIndex(test_y, prediction_tst.ravel())
#     combined_data_index = list(zip([ 'NSE','R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], index))
#     MLR.write_to_excel_file(['index','value'],combined_data_index , '.\output\SVM_rc_Index.xlsx', sheet_name='Sheet1')
#     print('index:', index)
# # ----------------- plot -------------------
#     # 设置图片的宽度和高度（单位为英寸）
#     width_inch = 40.0  # 设置图片的宽度为10英寸
#     height_inch = 6.0  # 设置图片的高度为6英寸
#
#     # 创建一个新的图像，并设置大小
#     plt.figure(figsize=(width_inch, height_inch))
#     plt.plot(t_for_training, LPJGUESSrunoff[:train_data_len], 'g', label='LPJGUESS_trn',linewidth=1.5)
#     plt.plot(t_for_training, train_y, 'b', label='GRDC_trn',linewidth=1.5)
#     plt.plot(t_for_training, train_y_r, 'r', label='Residual_trn', linewidth=1.5)
#     plt.plot(t_for_training, prediction_trn, 'y--', label='Prediction_trn',linewidth=1.5)
#
#
#     plt.plot(t_for_testing, LPJGUESSrunoff[train_data_len:], 'c', label='LPJGUESS_tst',linewidth=1.5)
#     plt.plot(t_for_testing, test_y, 'k', label='GRDC_tst',linewidth=1.5)
#     plt.plot(t_for_testing, test_y_r, 'r', label='Residual_tst', linewidth=1.5)
#     plt.plot(t_for_testing, prediction_tst, 'm--', label='Prediction_tst',linewidth=1.5)
#
#
#
#     plt.plot([t[train_data_len], t[train_data_len]], [-6, 6], 'r--', label='separation line',linewidth=1.5)  # separation line
#
#     plt.xlabel('Day')
#     plt.ylabel('Runoff')
#     plt.xlim(t[0], t[-1])
#     plt.ylim(-9, 9)
#     plt.legend(loc='upper right')
#     plt.text(10, 100, "train", size=15, alpha=1.0)
#     plt.text(data_len, 100, "test", size=15, alpha=1.0)
#     # 保存图像到文件
#     plt.savefig('.\output\SVM_residual_cliamte.png', dpi=600)
#     plt.show()
