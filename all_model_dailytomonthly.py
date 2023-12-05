import MLR
import pandas as pd
import MLR
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    # create database
    # 读取 Excel 文件

    df = pd.read_excel('.\output\LSTM\LSTM1966_2015_Random_iter5_ep200\LSTM_rc_Test.xlsx')
    ObsRunoff = df['GRDC_tst'].values  # 提取列数据到数组
    LSTMPreRunoff = df['Prediction_tst'].values
    ModelRunoff = df['LPJGUESS_tst'].values
    RFdailyindex=MLR.calIndex(ObsRunoff, LSTMPreRunoff)

    df = pd.read_excel(r'.\output\ANN\ANN1966_2015_r_Random_iter20_ep5\ann_rc_Test.xlsx')
    # ObsRunoff = df['GRDC_tst'].values  # 提取列数据到数组
    ANNPreRunoff = df['Prediction_tst'].values
    # ModelRunoff = df['LPJGUESS_tst'].values
    RFdailyindex = MLR.calIndex(ObsRunoff, ANNPreRunoff)

    df = pd.read_excel('.\output\RF\RF1966_2015_Random_iter20\RF_rc_Test.xlsx')
    #ObsRunoff = df['GRDC_tst'].values  # 提取列数据到数组
    RFPreRunoff = df['Prediction_tst'].values
    #ModelRunoff = df['LPJGUESS_tst'].values
    RFdailyindex = MLR.calIndex(ObsRunoff, RFPreRunoff)



    df = pd.read_excel(r'.\output\xgb\XGB1966_2015_Random_iter20\xgb_rc_Test.xlsx')
    #ObsRunoff = df['GRDC_tst'].values  # 提取列数据到数组
    XGBPreRunoff = df['Prediction_tst'].values
    #ModelRunoff = df['LPJGUESS_tst'].values
    RFdailyindex=MLR.calIndex(ObsRunoff, XGBPreRunoff)

    df = pd.read_excel('.\output\SVM\SVM1966_2015_r_Random_iter20\SVM_rc_Test.xlsx')
    #ObsRunoff = df['GRDC_tst'].values  # 提取列数据到数组
    SVMPreRunoff = df['Prediction_tst'].values
    #ModelRunoff = df['LPJGUESS_tst'].values
    RFdailyindex=MLR.calIndex(ObsRunoff, SVMPreRunoff)


    #print('index [NSE,R, PBIAS, RMSE, RMSEper, MAE, MAEper, PAE]:')
   # print('(ObsRunoff, PreRunoff)dailyindex:',dailyindex)
    #dailyindex1=MLR.calIndex(ObsRunoff, ModelRunoff)
    #print('(ObsRunoff, ModelRunoff)dailyindex',dailyindex1)




    data_len = ModelRunoff.shape[0]

    daynums = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    numofyear = int(ModelRunoff.shape[0] / 365);

    # 按照每个月的天数切分数据并存入数组
    daily_Obsrunoff = []
    daily_Modelrunoff = []

    LSTMdaily_Prerunoff = []
    RFdaily_Prerunoff = []
    ANNdaily_Prerunoff = []
    XGBdaily_Prerunoff = []
    SVMdaily_Prerunoff = []

    daily_Obsrunoff_cumulative=0
    daily_Modelrunoff_cumulative=0

    LSTMdaily_Prerunoff_cumulative = 0
    RFdaily_Prerunoff_cumulative=0
    ANNdaily_Prerunoff_cumulative=0
    XGBdaily_Prerunoff_cumulative=0
    SVMdaily_Prerunoff_cumulative=0


    start_index = 0
    #for yearid in np.linspace(1, numofyear, numofyear):
    for dayid in range(0,data_len):
        #end_index = start_index + num_days
        daily_Obsrunoff_cumulative += ObsRunoff[dayid]
        daily_Modelrunoff_cumulative += ModelRunoff[dayid]

        LSTMdaily_Prerunoff_cumulative += LSTMPreRunoff[dayid]
        RFdaily_Prerunoff_cumulative += RFPreRunoff[dayid]
        ANNdaily_Prerunoff_cumulative += ANNPreRunoff[dayid]
        XGBdaily_Prerunoff_cumulative += XGBPreRunoff[dayid]
        SVMdaily_Prerunoff_cumulative += SVMPreRunoff[dayid]

        daily_Obsrunoff.append(daily_Obsrunoff_cumulative )
        daily_Modelrunoff.append(daily_Modelrunoff_cumulative)

        LSTMdaily_Prerunoff.append(LSTMdaily_Prerunoff_cumulative)
        RFdaily_Prerunoff.append( RFdaily_Prerunoff_cumulative )
        ANNdaily_Prerunoff.append(ANNdaily_Prerunoff_cumulative)
        XGBdaily_Prerunoff.append(XGBdaily_Prerunoff_cumulative)
        SVMdaily_Prerunoff.append(SVMdaily_Prerunoff_cumulative)


            #start_index = end_index

    #RFmonthindex = calIndex(daily_Obsrunoff, RFdaily_Prerunoff)
    #ETmonthindex = calIndex(daily_Obsrunoff, ETdaily_Prerunoff)
    #GBmonthindex = calIndex(daily_Obsrunoff, GBdaily_Prerunoff)
    #SVMmonthindex = calIndex(daily_Obsrunoff, SVMdaily_Prerunoff)

    #combined_data_index = list(
    #    zip(['NSE', 'R', 'PBIAS', 'RMSE', 'RMSEper', 'MAE', 'MAEper', 'PAE'], RFmonthindex, ETmonthindex, GBmonthindex,
    #        SVMmonthindex))
    #write_to_excel_file(['index', 'RF','ET','GB','SVM'], combined_data_index, '.\output\only_ML\RF_ET_GB_SVM_Index_daily.xlsx',
    #                    sheet_name='Sheet1')

    # print('(ObsRunoff, PreRunoff)monthindex:', RFmonthindex)

    # monthindex1 = calIndex(daily_Obsrunoff, daily_Modelrunoff)
    # print('(ObsRunoff,ModelRunoff)monthindex:', monthindex1)

    # Use a classic, serif font
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12

    # Subtle color palette
    colors = ['navy', 'forestgreen', 'darkorange', 'maroon', 'steelblue', 'black', 'gray']
    #colors = ['#FFD92F', '#FF7F0E', '#2CA02C', '#98DF8A', '#C5B0D5', '#1F77B4', '#AEC7E8']

    # Create the plot
    t = np.linspace(1, data_len, data_len)
    plt.figure(figsize=(8, 6))  # Nature style often uses wider figures

    plt.plot(t, LSTMdaily_Prerunoff, label='LPJGUESS_LSTM', color=colors[0])
    plt.plot(t, RFdaily_Prerunoff, label='LPJGUESS_RF', color=colors[1])
    plt.plot(t, ANNdaily_Prerunoff, label='LPJGUESS_ANN', color=colors[2])
    plt.plot(t, XGBdaily_Prerunoff, label='LPJGUESS_XGB', color=colors[3])
    plt.plot(t, SVMdaily_Prerunoff, label='LPJGUESS_SVM', color=colors[4])

    plt.plot(t, daily_Obsrunoff, label='GRDC', color=colors[5], linestyle='--')
    plt.plot(t, daily_Modelrunoff, label='LPJGUESS', color=colors[6], linestyle=':')

    plt.xlabel('Time (day)')
    plt.ylabel('Runoff (km³)')

    plt.legend(loc='upper left', frameon=False)
    plt.xlim(0, data_len)
    plt.ylim(0, daily_Modelrunoff[data_len-1])

    # Save the figure in high resolution
    plt.savefig('.\output\model_cumulative_flow.png', dpi=1200, format='png')

    plt.show()
