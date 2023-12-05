import pandas as pd
import MLR
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

if __name__ == '__main__':
    # create database
    # 读取 Excel 文件
    modelname = 'XGB'
    path = f'.\output\Only ML'
    #path=f'.\output\{modelname}\SVM1966_2015_r_Random_iter20'

    df = pd.read_excel(f"{path}\{modelname}_c_Test.xlsx")

    # 提取两列数据到数组
    ObsRunoff = df['GRDC_tst'].values
    PreRunoff = df['Prediction_tst'].values
    ModelRunoff = df['LPJGUESS_tst'].values

    dailyindex=MLR.calIndex(ObsRunoff, PreRunoff)
    print('index [NSE,R, PBIAS, RMSE, RMSEper, MAE, MAEper, PAE]:')
    print('(ObsRunoff, PreRunoff)Dailyindex:',dailyindex)
    dailyindex1=MLR.calIndex(ObsRunoff, ModelRunoff)
    print('(ObsRunoff, ModelRunoff)Dailyindex',dailyindex1)


    from sklearn.feature_selection import mutual_info_regression

    def compute_mutual_information(series1, series2, max_lag, step=1):
        mi_scores = []
        lags = np.arange(-max_lag, max_lag + step, step)
        for lag in lags:
            lag = int(np.round(lag))  # 四舍五入到最接近的整数
            if lag < 0:
                shifted_series = np.roll(series2, -lag)
                mi = mutual_info_regression(series1[-lag:].reshape(-1, 1), shifted_series[-lag:])
            elif lag > 0:
                shifted_series = np.roll(series2, lag)
                mi = mutual_info_regression(series1[:-lag].reshape(-1, 1), shifted_series[:-lag])
            else:
                mi = mutual_info_regression(series1.reshape(-1, 1), series2)
            mi_scores.append(mi[0])  # mutual_info_regression returns an array
        return mi_scores, lags

    series1 = np.array(ObsRunoff)  #观测值
    series2 = np.array(ModelRunoff)  # 模型
    series3 = np.array(PreRunoff)  #预测值

    max_lag = 100  # 假设最大滞后为10

    mi_scores1, lags1 = compute_mutual_information(series1, series2, max_lag)

    # 找到最大的互信息值对应的滞后
    max_mi_lag1 = lags1[np.argmax(mi_scores1)]
    print(np.max(mi_scores1))
    print("Maximum mutual information advancing time for LPJ-GUESS:", max_mi_lag1)

    mi_scores2, lags2 = compute_mutual_information(series1, series3, max_lag)

    # 找到最大的互信息值对应的滞后
    max_mi_lag2 = lags2[np.argmax(mi_scores2)]
    print(np.max(mi_scores2))
    print(f"Maximum mutual information advancing time for LPJ-GUESS-{modelname}:", max_mi_lag2)

    combined_data_max_mi_lag = [[max_mi_lag1, max_mi_lag2]]

    columns_max_mi_lag = ['MMIDT for LPJ-GUESS', f'MMIDT for  Only {modelname}']

    MLR.write_to_excel_file(columns_max_mi_lag, combined_data_max_mi_lag, f'{path}\{modelname}_Maximum mutual information advancing time.xlsx', sheet_name = 'Sheet1')

    plt.figure(figsize=(5, 5))  # Wider figure

    # Plot the bars
    bar1=plt.bar(1, abs(max_mi_lag1), label='LPJGUESS', color='navy', width=0.4)
    bar2=plt.bar(2, abs(max_mi_lag2), label=f'Only {modelname}', color='forestgreen', width=0.4)
    for bar in bar1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{abs(max_mi_lag1)}', ha='center', va='bottom')
    for bar in bar2:
        if(max_mi_lag2==0):
            {
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), '<1', ha='center', va='bottom')
            }
        else:{
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{abs(max_mi_lag2)}', ha='center', va='bottom')
             }



    # Set the labels for the x-axis
    plt.xticks([1, 2], ['LPJ-GUESS', f'Only {modelname}'])

    plt.ylabel('Maximum mutual information advancing time (day)')
    plt.legend(loc='upper right', frameon=False)
    # Save the figure in high resolution
    plt.savefig(f'{path}\Maximum mutual information lag time.png', dpi=600, format='png')
    plt.show()




    data_len = ObsRunoff.shape[0]

    daynums=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    numofyear=int(ObsRunoff.shape[0]/365);

# 按照每个月的天数切分数据并存入数组
    monthly_Obsrunoff = []
    monthly_Prerunoff = []
    monthly_Modelrunoff = []
    start_index = 0
    for yearid in np.linspace(1, numofyear, numofyear):
        for num_days in daynums:
            end_index = start_index + num_days
            monthly_Obsrunoff.append(sum(ObsRunoff[start_index:end_index]))
            monthly_Prerunoff.append(sum(PreRunoff[start_index:end_index]))
            monthly_Modelrunoff.append(sum(ModelRunoff[start_index:end_index]))
            start_index = end_index

    monthindex = MLR.calIndex(monthly_Obsrunoff, monthly_Prerunoff)
    combined_data_index = list(zip(['NSE', 'R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], monthindex))
    MLR.write_to_excel_file(['index', 'value'], combined_data_index, f"{path}\{modelname}_Index_monthly.xlsx", sheet_name='Sheet1')

    print('(ObsRunoff, PreRunoff)monthindex:',monthindex)

    monthindex1=MLR.calIndex(monthly_Obsrunoff, monthly_Modelrunoff)
    print('(ObsRunoff,ModelRunoff)monthindex:',monthindex1)

    # Set a classic, serif font
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12

    # Create the plot with a subtle color palette
    t = np.linspace(1, 12 * numofyear, 12 * numofyear)
    plt.figure(figsize=(10, 5))  # Wider figure

    plt.plot(t, monthly_Prerunoff[:], label=f'Only {modelname}', color='darkorange')
    plt.plot(t, monthly_Obsrunoff[:], label='GRDC', color='navy')
    plt.plot(t, monthly_Modelrunoff[:], label='LPJGUESS', color='#AEC7E8')

    plt.xlabel('Month')
    plt.ylabel('Runoff (km³)')
    plt.legend(loc='upper right', frameon=False)

    # Save the figure in high resolution
    plt.savefig(f"{path}\{modelname}_climate_month.png", dpi=600, format='png')

    plt.show()

    MLR.scattarPlot(monthly_Obsrunoff, monthly_Prerunoff,f'Only {modelname}',200, path)
    MLR.scattarPlot(monthly_Obsrunoff, monthly_Modelrunoff, 'LPJGUESS',200, path)



    # # 示例时间序列
    # series1 = np.array(monthly_Obsrunoff)  # 预测值
    # series2 = np.array(monthly_Modelrunoff)  # 观测值
    #
    # # 计算交叉相关
    # correlation = signal.correlate(series2, series1, mode='full')
    #
    # # 计算滞后
    # lags = signal.correlation_lags(len(series2), len(series1), mode='full')
    # max_lag1 = lags[np.argmax(np.abs(correlation))]
    # print(np.abs(correlation))
    # print(np.argmax(np.abs(correlation)))
    # print("最大交叉相关系数对应的滞后时间:", max_lag1)
    #
    # # 示例时间序列
    # series1 = np.array(monthly_Obsrunoff)  # 预测值
    # series3 = np.array(monthly_Prerunoff)  # 观测值
    #
    # # 计算交叉相关
    # correlation2 = signal.correlate(series3, series1, mode='full')
    #
    # # 计算滞后
    # lags2 = signal.correlation_lags(len(series3), len(series1), mode='full')
    # max_lag2 = lags2[np.argmax(np.abs(correlation))]
    #
    # print("最大交叉相关系数对应的滞后时间:", max_lag2)
