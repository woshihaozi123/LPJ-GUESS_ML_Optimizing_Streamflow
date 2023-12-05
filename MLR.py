from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv;
import pandas as pd
import xlrd
import xlwt
import numpy as np
import math
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import xgboost as xgb



def read_excel_file(file_path, sheet_name='Sheet1'):
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        column_names = list(data.columns)
        return column_names, data
    except Exception as e:
        print(f"Error occurred while reading the Excel file: {e}")
        return None, None

def write_to_excel_file(column_names, data, file_path, sheet_name='Sheet1'):
    try:
        df = pd.DataFrame(data, columns=column_names)
        df.to_excel(file_path, sheet_name=sheet_name, index=False)
        print(f"Data written to Excel file: {file_path}")
    except Exception as e:
        print(f"Error occurred while writing to the Excel file: {e}")

def set_style(name, height, bold=False):
    style = xlwt.XFStyle()   # 初始化样式
    font = xlwt.Font()       # 为样式创建字体
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def writeexcel(path,result):
    # 创建工作簿
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建sheet
    data_sheet = workbook.add_sheet('result')

    # 生成第一行和第二行
    if(isinstance(result,np.ndarray)==True):
        for i in range(len(result)):
                data_sheet.write(i,0,result[i] , set_style('Times New Roman', 220, True))
    elif(isinstance(result,list)==True):
         for i in range(len(result)):
                data_sheet.write(i,0,result[i] , set_style('Times New Roman', 220, True))
    else:#numpy.float64
         data_sheet.write(0,0,result.astype(type('float', (float,), {})), set_style('Times New Roman', 220, True))
    # 保存文件
    workbook.save(path)

def readexcel(InputFile_path):
    X=[]
    y=[]
    # 设置路径

    # 打开execl
    workbook = xlrd.open_workbook(InputFile_path)
    # 输出Excel文件中所有sheet的名字
    print(workbook.sheet_names())

    #根据sheet索引或者名称获取sheet内容
    #Data_sheet = workbook.sheets()[0]  # 通过索引获取
    Data_sheet = workbook.sheet_by_index(0)  # 通过索引获取
    # Data_sheet = workbook.sheet_by_name(u'名称')  # 通过名称获取

    print(Data_sheet.name)  # 获取sheet名称
    rowNum = Data_sheet.nrows  # sheet行数
    colNum = Data_sheet.ncols  # sheet列数
    print(rowNum)
    print(colNum)
    # 获取所有单元格的内容

    for i in range(rowNum):
        rowlist = []
        for j in range(colNum):
            if(j<colNum-1):
                rowlist.append(Data_sheet.cell_value(i, j))
            else:
                y.append(Data_sheet.cell_value(i, j))
        X.append(rowlist)
    return X,y



def calIndex(obs, pre):
    if len(obs) != len(pre):
        print("calIndex: input error!")

    avey = sum(obs) / len(obs)

    sumRMSE = 0
    for i in range(len(obs)):
        sumRMSE += math.pow((obs[i] - pre[i]), 2) / len(obs)
    RMSE = math.pow(sumRMSE, 0.5)
    RMSEper = RMSE / avey

    MAE = sum(abs(obs[i] - pre[i]) for i in range(len(obs))) / len(obs)

    MAEper = sum(abs((obs[i] - pre[i]) / obs[i]) for i in range(len(obs)) if obs[i] != 0) / len(obs)

    PAE = sum((1 - abs((obs[i] - pre[i]) / obs[i])) for i in range(len(obs)) if obs[i] != 0) / len(obs)

    SSE = sum(math.pow((obs[i] - pre[i]), 2) for i in range(len(obs)))
    SST = sum(math.pow((obs[i] - avey), 2) for i in range(len(obs)))
    R2 = 1 - (SSE / SST)

    # Calculate NSE (Nash-Sutcliffe efficiency)
    numerator_nse = sum(math.pow((obs[i] - pre[i]), 2) for i in range(len(obs)))
    denominator_nse = sum(math.pow((obs[i] - avey), 2) for i in range(len(obs)))
    NSE = 1 - (numerator_nse / denominator_nse)

    # Calculate Percent Bias (PBIAS)
    PBIAS = 100 * sum((pre[i] - obs[i]) for i in range(len(obs))) / sum(obs)

    # Calculate Pearson correlation coefficient (r)
    R = np.corrcoef(obs, pre)[0, 1]
    return [NSE,R,PBIAS, RMSE, RMSEper, MAE, MAEper, PAE]

def calIndex1(y,y1):
    if(len(y)!=len(y1)):
        print("calIndex: input error!")
    avey=0
    sumy=0
    n=len(y)
    for i in range(n):
        sumy+=y[i]
    avey=sumy/n

    sumRMSE=0
    for i in range(n):
        sumRMSE+=math.pow((y[i]-y1[i]),2)/n
    RMSE=math.pow(sumRMSE,0.5)
    RMSEper=RMSE/avey

    MAE=0
    for i in range(n):
        MAE+=abs((y[i]-y1[i]))/n

    MAEper=0
    for i in range(n):
        if(y[i]==0):continue
        MAEper+=abs((y[i]-y1[i]))/n/y[i]

    P=0
    for i in range(n):
       if(y[i]==0):continue
       P+= (1-abs((y[i]-y1[i])/y[i]))/n
    R2=0
    SSE=0
    SST=0
    for i in range(n):
        SSE+=math.pow((y[i]-y1[i]),2)
        SST+=math.pow((y[i]-avey),2)
    R2=1-(SSE/SST)

    return [R2,RMSE,RMSEper,MAE,MAEper,P]

def calR2(y,y1):
    avey=0
    sumy=0
    n=len(y)
    for i in range(n):
        sumy+=y[i]
    avey=sumy/n
    R2=0
    SSE=0
    SST=0
    for i in range(n):
        SSE+=math.pow((y[i]-y1[i]),2)
        SST+=math.pow((y[i]-avey),2)
    R2=1-(SSE/SST)
    return R2

def XGB(train_x, train_y,test_x,test_y):
    # 定义超参数范围
    xgb_params = {
        'max_depth': sp_randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': sp_randint(100, 1000),
        'gamma': uniform(0, 0.5),
        'min_child_weight': sp_randint(1, 10),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5)
    }

    # Create a XGB Regressor object
    clf_xgb = xgb.XGBRegressor(objective='reg:squarederror')

    # Optimize hyperparameters using Randomized Search
    Random_search = RandomizedSearchCV(clf_xgb, param_distributions=xgb_params, n_iter=20, cv=5,
                                       scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
    Random_search.fit(train_x, train_y)

    # save best_params into Excel file
    best_params = Random_search.best_params_
    print("best_params:", best_params)

    # convert best_params to DataFrame
    df_best_params = pd.DataFrame([best_params])

    # save into Excel file
    excel_filename = '.\output\XGBoost_best_params.xlsx'
    df_best_params.to_excel(excel_filename, index=False)

    print(f"best_params save into Excel file {excel_filename}")
    # Make predictions
    predictive_y_for_training = Random_search.predict(train_x)
    predictive_y_for_testing = Random_search.predict(test_x)

    # ----------------- save data -------------------
    prediction_trn = predictive_y_for_training
    prediction_tst = predictive_y_for_testing
    # Combine the lists into four columns
    combined_data_trn = list(zip(LPJGUESSrunoff[:train_data_len], train_y.ravel(), train_y.ravel(), prediction_trn))
    combined_data_tst = list(zip(LPJGUESSrunoff[train_data_len:], test_y.ravel(), test_y.ravel(), prediction_tst))
    columns_trn = ['LPJGUESS_trn', 'GRDC_trn', 'GRDC_trn', 'Prediction_trn']
    columns_tst = ['LPJGUESS_tst', 'GRDC_tst', 'GRDC_tst', 'Prediction_tst']
    write_to_excel_file( columns_trn, combined_data_trn, r'.\output\XGB_c_Training.xlsx', sheet_name = 'Sheet1')
    write_to_excel_file(columns_tst, combined_data_tst, r'.\output\XGB_c_Test.xlsx', sheet_name='Sheet1')
    index=calIndex(test_y.ravel(), prediction_tst.ravel())
    combined_data_index = list(zip([ 'NSE','R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], index))
    write_to_excel_file(['index','value'],combined_data_index , r'.\output\XGB_c_Index.xlsx', sheet_name='Sheet1')
    print('index:', index)

    # ----------------- plot -------------------
    # 设置图片的宽度和高度（单位为英寸）
    width_inch = 40.0  # 设置图片的宽度为10英寸
    height_inch = 6.0  # 设置图片的高度为6英寸

    # 创建一个新的图像，并设置大小
    plt.figure(figsize=(width_inch, height_inch))
    plt.plot(t_for_training, LPJGUESSrunoff[:train_data_len].ravel(), 'g', label='LPJGUESS_trn', linewidth=1.5)
    plt.plot(t_for_training, train_y.ravel(), 'b', label='GRDC_trn', linewidth=1.5)
    plt.plot(t_for_training, prediction_trn.ravel(), 'y--', label='Prediction_trn', linewidth=1.5)

    plt.plot(t_for_testing, LPJGUESSrunoff[train_data_len:].ravel(), 'c', label='LPJGUESS_tst', linewidth=1.5)
    plt.plot(t_for_testing, test_y.ravel(), 'k', label='GRDC_tst', linewidth=1.5)
    plt.plot(t_for_testing, prediction_tst.ravel(), 'm--', label='Prediction_tst', linewidth=1.5)

    plt.plot([t[train_data_len], t[train_data_len]], [-6, 6], 'r--', label='separation line',
             linewidth=1.5)  # separation line

    plt.xlabel('Day')
    plt.ylabel('Runoff')
    plt.xlim(t[0], t[-1])
    plt.ylim(-9, 9)
    plt.legend(loc='upper right')
    plt.text(10, 100, "train", size=15, alpha=1.0)
    plt.text(data_len, 100, "test", size=15, alpha=1.0)
    # 保存图像到文件
    plt.savefig(r'.\output\XGB_cliamte.png', dpi=600)
    plt.show()
    return index,predictive_y_for_testing

def RandomForest(train_x, train_y,test_x,test_y):
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
    Random_search.fit(train_x, train_y)

    # save best_params into Excel file
    best_params = Random_search.best_params_
    print("best_params:", best_params)

    # convert best_params to DataFrame
    df_best_params = pd.DataFrame([best_params])

    # save into Excel file
    excel_filename = '.\output\RF_best_params.xlsx'
    df_best_params.to_excel(excel_filename, index=False)

    # Make predictions
    predictive_y_for_training = Random_search.predict(train_x)
    predictive_y_for_testing = Random_search.predict(test_x)

    # ----------------- save data -------------------
    prediction_trn = predictive_y_for_training
    prediction_tst = predictive_y_for_testing
    # Combine the lists into four columns
    combined_data_trn = list(zip(LPJGUESSrunoff[:train_data_len], train_y.ravel(), train_y.ravel(), prediction_trn.ravel()))
    combined_data_tst = list(zip(LPJGUESSrunoff[train_data_len:], test_y.ravel(), test_y.ravel(), prediction_tst.ravel()))
    columns_trn = ['LPJGUESS_trn', 'GRDC_trn', 'GRDC_trn', 'Prediction_trn']
    columns_tst = ['LPJGUESS_tst', 'GRDC_tst', 'GRDC_tst', 'Prediction_tst']
    write_to_excel_file(columns_trn, combined_data_trn, '.\output\RF_c_Training.xlsx', sheet_name='Sheet1')
    write_to_excel_file(columns_tst, combined_data_tst, '.\output\RF_c_Test.xlsx', sheet_name='Sheet1')
    index = calIndex(test_y.ravel(), prediction_tst.ravel())
    combined_data_index = list(zip(['NSE', 'R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], index))
    write_to_excel_file(['index', 'value'], combined_data_index, '.\output\RF_c_Index.xlsx', sheet_name='Sheet1')
    print('index:', index)
    # ----------------- plot -------------------
    # 设置图片的宽度和高度（单位为英寸）
    width_inch = 40.0  # 设置图片的宽度为10英寸
    height_inch = 6.0  # 设置图片的高度为6英寸

    # 创建一个新的图像，并设置大小
    plt.figure(figsize=(width_inch, height_inch))
    plt.plot(t_for_training, LPJGUESSrunoff[:train_data_len].ravel(), 'g', label='LPJGUESS_trn', linewidth=1.5)
    plt.plot(t_for_training, train_y.ravel(), 'b', label='GRDC_trn', linewidth=1.5)
    #plt.plot(t_for_training, train_y, 'r', label='Residual_trn', linewidth=1.5)
    plt.plot(t_for_training, prediction_trn.ravel(), 'y--', label='Prediction_trn', linewidth=1.5)

    plt.plot(t_for_testing, LPJGUESSrunoff[train_data_len:].ravel(), 'c', label='LPJGUESS_tst', linewidth=1.5)
    plt.plot(t_for_testing, test_y.ravel(), 'k', label='GRDC_tst', linewidth=1.5)
    #plt.plot(t_for_testing, test_y, 'r', label='Residual_tst', linewidth=1.5)
    plt.plot(t_for_testing, prediction_tst.ravel(), 'm--', label='Prediction_tst', linewidth=1.5)

    plt.plot([t[train_data_len], t[train_data_len]], [-6, 6], 'r--', label='separation line',
             linewidth=1.5)  # separation line

    plt.xlabel('Day')
    plt.ylabel('Runoff')
    plt.xlim(t[0], t[-1])
    plt.ylim(-9, 9)
    plt.legend(loc='upper right')
    plt.text(10, 100, "train", size=15, alpha=1.0)
    plt.text(data_len, 100, "test", size=15, alpha=1.0)
    # 保存图像到文件
    plt.savefig('.\output\RF_cliamte.png', dpi=600)
    plt.show()
    return index, predictive_y_for_testing

def SVR(train_x, train_y,test_x,test_y):
    # ----------------- train -------------------
    # Define SVM Hyperparameter configuration space
    svm_params = {
        'C': uniform(0, 50),
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': uniform(0.1, 1)  # 添加gamma参数
    }

    # Create support vector machine objects
    clf = svm.SVR()

    # Optimize hyperparameters using random search
    Random_search = RandomizedSearchCV(clf, param_distributions=svm_params, n_iter=20, cv=5,
                                       # scoring='r2')
                                       scoring='neg_mean_squared_error', n_jobs=-1)
    Random_search.fit(train_x, train_y)

    # save best_params into Excel file
    best_params = Random_search.best_params_
    print("best_params:", best_params)

    # convert best_params to DataFrame
    df_best_params = pd.DataFrame([best_params])

    # save into Excel file
    excel_filename = '.\output\SVM_best_params.xlsx'
    df_best_params.to_excel(excel_filename, index=False)

    print(f"best_params save into Excel file {excel_filename}")

    predictive_y_for_training = Random_search.predict(train_x)
    predictive_y_for_testing = Random_search.predict(test_x)

    # ----------------- save data -------------------
    prediction_trn = predictive_y_for_training
    prediction_tst = predictive_y_for_testing
    # Combine the lists into four columns
    combined_data_trn = list(zip(LPJGUESSrunoff[:train_data_len], train_y.ravel(), train_y.ravel(), prediction_trn.ravel()))
    combined_data_tst = list(zip(LPJGUESSrunoff[train_data_len:], test_y.ravel(), test_y.ravel(), prediction_tst.ravel()))
    columns_trn = ['LPJGUESS_trn', 'GRDC_trn', 'GRDC_trn', 'Prediction_trn']
    columns_tst = ['LPJGUESS_tst', 'GRDC_tst', 'GRDC_tst', 'Prediction_tst']
    write_to_excel_file(columns_trn, combined_data_trn, '.\output\SVM_c_Training.xlsx', sheet_name='Sheet1')
    write_to_excel_file(columns_tst, combined_data_tst, '.\output\SVM_c_Test.xlsx', sheet_name='Sheet1')
    index = calIndex(test_y.ravel(), prediction_tst.ravel())
    combined_data_index = list(zip(['NSE', 'R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], index))
    write_to_excel_file(['index', 'value'], combined_data_index, '.\output\SVM_c_Index.xlsx', sheet_name='Sheet1')
    print('index:', index)
    # ----------------- plot -------------------
    # 设置图片的宽度和高度（单位为英寸）
    width_inch = 40.0  # 设置图片的宽度为10英寸
    height_inch = 6.0  # 设置图片的高度为6英寸

    # 创建一个新的图像，并设置大小
    plt.figure(figsize=(width_inch, height_inch))
    plt.plot(t_for_training, LPJGUESSrunoff[:train_data_len].ravel(), 'g', label='LPJGUESS_trn', linewidth=1.5)
    plt.plot(t_for_training, train_y.ravel(), 'b', label='GRDC_trn', linewidth=1.5)
    #plt.plot(t_for_training, train_y_r, 'r', label='Residual_trn', linewidth=1.5)
    plt.plot(t_for_training, prediction_trn.ravel(), 'y--', label='Prediction_trn', linewidth=1.5)

    plt.plot(t_for_testing, LPJGUESSrunoff[train_data_len:].ravel(), 'c', label='LPJGUESS_tst', linewidth=1.5)
    plt.plot(t_for_testing, test_y.ravel(), 'k', label='GRDC_tst', linewidth=1.5)
    #plt.plot(t_for_testing, test_y, 'r', label='Residual_tst', linewidth=1.5)
    plt.plot(t_for_testing, prediction_tst.ravel(), 'm--', label='Prediction_tst', linewidth=1.5)

    plt.plot([t[train_data_len], t[train_data_len]], [-6, 6], 'r--', label='separation line',
             linewidth=1.5)  # separation line

    plt.xlabel('Day')
    plt.ylabel('Runoff')
    plt.xlim(t[0], t[-1])
    plt.ylim(-9, 9)
    plt.legend(loc='upper right')
    plt.text(10, 100, "train", size=15, alpha=1.0)
    plt.text(data_len, 100, "test", size=15, alpha=1.0)
    # 保存图像到文件
    plt.savefig('.\output\SVM_cliamte.png', dpi=600)
    plt.show()

    return index, predictive_y_for_testing

def ExtraTrees(train_x,train_y,test_x,test_y):

    regr = ExtraTreesRegressor(n_estimators=1000,max_features=1.0,oob_score=True,bootstrap=True)
    regr.fit(train_x, train_y)
    pre_y = regr.predict(test_x)
    index = calIndex(test_y, pre_y)
    return index,pre_y


    # print("regr.feature_importances_")
    # print(regr.feature_importances_)
    # writeexcel(r"C:/Users/周昊/Desktop/IJGIS paper/七改/数据/机器学习/workdays_ExtraTrees_feature_importances_.xls",regr.feature_importances_)
    # print("regr.oob_score_")
    # print(regr.oob_score_)
    # #writeexcel(r"F:/机器学习方法/data/RT4_holi1_ExtraTrees_feature_oob_score_.xls",regr.oob_score_)
    # #print("regr.oob_prediction_")
    # #print(regr.oob_prediction_)
    # #writeexcel(r"F:/机器学习方法/data/RT4_holi1_RandomForest_oob_prediction_.xls",regr.oob_prediction_)
    #
    # y1=regr.predict(X)
    # index=[]
    # index=calIndex(y,y1)
    #
    # print(index)
    # writeexcel(r"C:/Users/周昊/Desktop/IJGIS paper/七改/数据/机器学习/workdays_ExtraTrees_index.xls",index)
    # R2=calR2(y,y1)
    # return R2

def GradientBoost(train_x,train_y,test_x,test_y):
    regr = GradientBoostingRegressor(loss = 'quantile',alpha=0.3,max_depth=6)
    regr.fit(train_x, train_y)
    pre_y = regr.predict(test_x)
    index = calIndex(test_y, pre_y)
    return index,pre_y

def MLP(train_x,train_y,test_x,test_y):
    regr = MLPRegressor(solver='lbfgs')
    regr.fit(train_x, train_y)
    pre_y = regr.predict(test_x)
    index = calIndex(test_y, pre_y)
    return index,pre_y



def KNN(train_x,train_y,test_x,test_y):
    regr = KNeighborsRegressor(n_neighbors=5)
    regr.fit(train_x, train_y)
    pre_y = regr.predict(test_x)
    index = calIndex(test_y, pre_y)
    return index,pre_y

def calrelativeimportances(fun,X,y):
    r2list=[]
    n=len(X)
    m=len(X[0])
    print(n,m)

    R2last=0
    R2now=0

    for i in range(m):
        X1=np.zeros((n,m-i))
        #print(X[0:][0])
        #print(len(X[:][0:m-1-i]),len(X[:][0:m-1-i][0]))

        for a in range(n):
            for b in range(m-i):
                #print(a,b)
                X1[a][b]=X[a][b]

        R2now=fun(X1,y)

        if(i>0):
            r2list.insert(0,R2last-R2now)
        R2last=R2now

    #加上R2 of first factor
    R2=fun(X,y)
    X2=np.zeros((n,m-1))
    for a in range(n):
        for b in range(m):
            if(b>0):
                X2[a][b-1]=X[a][b]


    R2_deletefirstfactor=fun(X2,y)
    r2list.insert(0,R2-R2_deletefirstfactor)

    #normal
    maxR2=max(r2list)
    minR2=min(r2list)

    r2list1=[]
    r2list1 = list(map(lambda x: (x-minR2)/(maxR2-minR2),r2list))


    print(fun.__name__+"relative importance")
    print(r2list1)
    writeexcel(r"F:/机器学习方法/data/RT4_notholi1_"+fun.__name__+"_feature_importances.xls",r2list1)

def scattarPlot(obs, pre,titlestr,xylim,path):
    # Set the classic font
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12

    plt.figure(figsize=(6, 6))  # Square aspect ratio
    # 绘制散点图
    plt.scatter(obs[:], pre[:],s=10, color='navy')
    plt.plot([0,xylim],[0,xylim])
    plt.xlim(0,xylim)
    plt.ylim(0, xylim)
    index=calIndex(obs, pre)
    print('index:',index)
    # 标注指标
    # 设置R2值和NSE值

    # 设置文本框的边框样式和背景颜色
    bbox_style = dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white')
    #[NSE,R, PBIAS, RMSE, RMSEper, MAE, MAEper, PAE]
    #nse_value =index[0]  # 假设NSE值为0.87
    r_value = index[1]  # 假设R2值为0.95
    PBIAS_value=index[2]
    RMSE_value=index[3]
    MAE_value=index[5]
    PAE_value=index[7]
    plt.text(plt.xlim()[1]-10, plt.ylim()[0]+10, f"RMSE={RMSE_value:.2f}\nMAE={MAE_value:.2f}\nR = {r_value:.2f}\nPAE={PAE_value:.2f}\nPBIAS = {PBIAS_value:.2f}", fontsize=12, ha='right',
             va='baseline',bbox=bbox_style)

    # 设置图表标题和标签
    plt.title(titlestr)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')

    # 显示图例
    #plt.legend()

    plt.savefig(f'{path}\{titlestr}'+'.png', dpi=1200, format='png')
    # 展示图表
    plt.show()



if __name__ == '__main__':

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

    #LPJGUESSrunoff_normalized = scaler.fit_transform(LPJGUESSrunoff.reshape(-1, 1))

    ###----------------- add all variables into a datasets for uaage-----------------
    data_len = LPJGUESSrunoff.shape[0]
    t = np.linspace(1, data_len, data_len)
    dataset = np.zeros((data_len, 8 + 6))

    # X variables
    dataset[:, 0] = LPJGUESSrunoff.reshape(-1, 1)
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
    model_y=dataset[train_data_len:, 0]

    # -----------------INPUT and output FEATURES number-----------------
    INPUT_FEATURES_NUM = 11
    OUTPUT_FEATURES_NUM = 1




    RFindex,RFpre_y=RandomForest(train_x,train_y,test_x,test_y)
    XGBindex, XGBpre_y = XGB(train_x, train_y, test_x, test_y)
    SVMindex,SVMpre_y=SVR(train_x,train_y,test_x,test_y)

    #ETindex, ETpre_y = ExtraTrees(train_x, train_y, test_x, test_y)
    #GBindex, GBpre_y = GradientBoost(train_x, train_y, test_x, test_y)
    #MLPindex,MLPpre_y=MLP(train_x,train_y,test_x,test_y)
    #KNNindex,KNNpre_y=KNN(train_x,train_y,test_x,test_y)

    #calrelativeimportances(GradientBoost,X,y)
    #calrelativeimportances(MLP,X,y)
   # calrelativeimportances(SVR,X,y)
   # calrelativeimportances(KNN,X,y)


    data_len = test_x.shape[0]

    daynums = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    numofyear = int(test_x.shape[0] / 365)

    # 按照每个月的天数切分数据并存入数组
    monthly_Obsrunoff = []
    RFmonthly_Prerunoff = []
    #ETmonthly_Prerunoff = []
    XGBmonthly_Prerunoff = []
    SVMmonthly_Prerunoff = []

    monthly_Modelrunoff = []
    start_index = 0
    for yearid in np.linspace(1, numofyear, numofyear):
        for num_days in daynums:
            end_index = start_index + num_days
            monthly_Obsrunoff.append(sum(test_y[start_index:end_index]))
            RFmonthly_Prerunoff.append(sum(RFpre_y[start_index:end_index]))
            #ETmonthly_Prerunoff.append(sum(ETpre_y[start_index:end_index]))
            XGBmonthly_Prerunoff.append(sum(XGBpre_y[start_index:end_index]))
            SVMmonthly_Prerunoff.append(sum(SVMpre_y[start_index:end_index]))

            monthly_Modelrunoff.append(sum(model_y[start_index:end_index]))
            start_index = end_index

    RFmonthindex = calIndex(monthly_Obsrunoff, RFmonthly_Prerunoff)
    #ETmonthindex = calIndex(monthly_Obsrunoff, ETmonthly_Prerunoff)
    XGBmonthindex = calIndex(monthly_Obsrunoff, XGBmonthly_Prerunoff)
    SVMmonthindex = calIndex(monthly_Obsrunoff, SVMmonthly_Prerunoff)

    combined_data_index = list(
        zip(['NSE', 'R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], RFmonthindex, XGBmonthindex,
            SVMmonthindex))
    write_to_excel_file(['index', 'RF','XGB','SVM'], combined_data_index, '.\output\Only ML\RF_XGB_SVM_Index_monthly.xlsx',
                        sheet_name='Sheet1')

    # print('(ObsRunoff, PreRunoff)monthindex:', RFmonthindex)

    # monthindex1 = calIndex(monthly_Obsrunoff, monthly_Modelrunoff)
    # print('(ObsRunoff,ModelRunoff)monthindex:', monthindex1)

    t = np.linspace(1, 12 * numofyear, 12 * numofyear)

    plt.figure()
    plt.plot(t, RFmonthly_Prerunoff[:], label='RF_only')
    #plt.plot(t, ETmonthly_Prerunoff[:], label='ET_only')
    plt.plot(t, XGBmonthly_Prerunoff[:], label='XGB_only')
    plt.plot(t, SVMmonthly_Prerunoff[:], label='SVM_only')

    plt.plot(t, monthly_Obsrunoff[:], label='GRDC')
    #plt.plot(t, monthly_Modelrunoff[:], label='LPJGUESS')
    # plt.plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = 2.5')  # t = 2.5
    # plt.plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t = 6.8')  # t = 6.8
    plt.xlabel('month')
    # plt.ylim(-1.2, 1.2)
    plt.ylabel('Runoff(km$^3$)')
    plt.legend(loc='upper right')
    plt.savefig('.\output\Only ML\RF_ET_GB_SVM_cliamte_month.png', dpi=1200)
    plt.show()
    scattarPlot(monthly_Obsrunoff, RFmonthly_Prerunoff, 'RF_only', 200,'.\output\Only ML')
    #scattarPlot(monthly_Obsrunoff, ETmonthly_Prerunoff, 'ET_only', 200)
    scattarPlot(monthly_Obsrunoff, XGBmonthly_Prerunoff, 'XGB_only', 200,'.\output\Only ML')
    scattarPlot(monthly_Obsrunoff, SVMmonthly_Prerunoff, 'SVM_only', 200,'.\output\Only ML')