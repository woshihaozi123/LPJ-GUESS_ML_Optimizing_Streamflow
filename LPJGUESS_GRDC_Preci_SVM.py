import pandas as pd
import MLR
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    # create database
    # 读取 Excel 文件

# create database
    # 读取 Excel 文件
    df = pd.read_excel('.\data\RunoffDailySeries20002015.xlsx')

    # 提取两列数据到数组
    LPJGUESSrunoff = df['LPJRunoff'].values
    GRDCrunoff = df['GRDCRunoff'].values
    Residualrunoff=df['Residual'].values

    df = pd.read_excel('.\data\ClimateDailySeries20002015.xlsx')
    Temp= df['Temp'].values
    Prec = df['Prec'].values
    Rad = df['Rad'].values
    U10 = df['U10'].values
    Relhum = df['Relhum'].values

    data_len = LPJGUESSrunoff.shape[0]



    numofyear = int(LPJGUESSrunoff.shape[0] / 365);


    # 按照每个月的天数切分数据并存入数组
    yearly_Obsrunoff = []
    yearly_Precipiation= []
    yearly_Modelrunoff = []

    for yearid in range(0, numofyear):
        Obsrunoff = 0
        Precipiation = 0
        Modelrunoff = 0
        for dayid in range(0,365):
            Obsrunoff += GRDCrunoff[yearid*365+dayid]
            Precipiation += Prec[yearid*365+dayid]
            Modelrunoff += LPJGUESSrunoff[yearid*365+dayid]

        yearly_Obsrunoff.append(Obsrunoff)
        yearly_Precipiation.append(Precipiation)
        yearly_Modelrunoff.append(Modelrunoff)

    # Generate the array t using np.linspace
    t = np.linspace(2000, 2000 + numofyear - 1, numofyear)

    # Convert the elements of t to integers
    t = t.astype(int)

    # Set the font size and style for the plot
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
    })

    # Create the figure and plot the data
    plt.figure(figsize=(8, 6))  # Set the figure size (width, height) in inches

    # Plot the data with line plots
    plt.plot(t, yearly_Obsrunoff, label='Observation', linestyle='-', marker='o', color='blue')
    plt.plot(t, yearly_Modelrunoff, label='LPJGUESS', linestyle='-', marker='s', color='green')

    # Plot the data with a bar plot
    plt.bar(t, yearly_Precipiation, label='Precipitation', color='gray', alpha=0.5)

    # Set the x and y-axis labels
    plt.xlabel('Year')
    plt.ylabel('Volume of Water ($\mathregular{km^3}$)')  # Use LaTeX notation for the exponent

    # Set the x-axis ticks and labels
    plt.xticks(t, rotation=45, ha='right')

    # Add a legend in the upper left corner
    plt.legend(loc='upper left')

    # Save the plot as an image file with high resolution (dpi=1200)
    plt.savefig('.\output\LPJ_GRDC_Preci.png', dpi=1200)

    # Display the plot
    plt.show()

