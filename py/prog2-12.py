# プログラム2.12

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# データの準備
def prepare():
    # 世界の気温を記録したデータをダウンロードして読み込む
    #!kaggle datasets download -d schedutron/global-temperatures
    get_ipython().system('unzip global-temperatures.zip')
    df = pd.read_csv('GlobalTemperatures.csv')
    # 平均気温をデータフレームに格納する
    df = pd.DataFrame(df, columns=['dt', 'LandAverageTemperature'])
    print(df)
    return df

# 欠損値を含む区間および欠損値が最も多い区間を確認
def get_missing_range(df):
    i = 0
    n = 20
    rlist = []
    cmax = 0
    ci = -1
    while i+n < len(df):
        mc = df[i:i+n].isnull().sum(axis=1)
        c = np.sum(mc.values)
        if c > 0:
            rlist.append(range(i,i+n))
            if cmax < c:
                cmax = c
                ci = len(rlist)-1
        i += n
    return rlist, ci

# 欠損値が最も多い区間について図示
def check_missing_values(df):
    plt.figure(figsize=(7,5))
    df['dt'] = pd.to_datetime(df['dt'])
    df.index = pd.DatetimeIndex(df['dt'], name='Date')
    df.drop('dt', axis=1, inplace=True)
    # 欠損値のある範囲を確認
    rlist, ci = get_missing_range(df)
    print('-->', list(rlist[ci]))
    l = list(rlist[ci])
    print('Include missing value', df[l[0]:l[-1]])
    plt.plot(df[l[0]:l[-1]]) 
    plt.xticks(rotation=90)
    plt.title('Including Missing Values')
    plt.xlabel('Datetime')
    plt.ylabel('Temperature')
    plt.savefig('including_missing_value.png', bbox_inches='tight') 
    plt.show()
    return rlist[ci]

# 欠損値の補間（interpolateメソッドを利用)
def interpolate(df, itype, direction, a_range):
    print('{} interpolate'.format(itype))
    if itype == None:
      df_i = df.interpolate(limit=1,
            limit_direction=direction)['LandAverage Temperature']
    else:
      df_i = df.interpolate(method=itype, order=1)['LandAverageTemperature']
    print(df_i[list(a_range)]) 
    plt.figure(figsize=(7,5))
    plt.plot(df_i[list(a_range)])
    plt.title('{} interpolation'.format(itype))
    plt.xticks(rotation=90)
    plt.xlabel('Datetime')
    plt.ylabel('Temperature')
    plt.savefig('{}_interpolation.png'.format(itype), bbox_inches='tight')
    plt.show()

def main():
    df = prepare()
    a_range = check_missing_values(df)
    direction = 'forward'
    for itype in ['time', 'values', 'linear',
                  'index', 'spline', 'nearest']:
        interpolate(df, itype, direction, a_range)

if __name__ == '__main__':
    main()

