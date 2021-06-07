# プログラム2.8

import pandas as pd
import numpy as np
# 記録された時刻を時系列として扱うために必要
from datetime import datetime

# データの準備
def prepare():
    #!kaggle datasets download -d jsphyg/weather-dataset-rattle-package
    get_ipython().system('unzip weather-dataset-rattle-package.zip')
    
# 前処理（欠損値の削除など）
def preprocess():
    df = pd.read_csv('weatherAUS.csv')
    df = df.replace('NA', 'NaN')
    print(df)
    df = df.dropna(how='any')
    features = ['Date', 'MinTemp', 'MaxTemp', 'Rainfall']
    # datetime型に変換する
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    # Yes/No ==> 1/0 に変換
    df['RainTomorrow'] = df['RainTomorrow'].map(
                          {'No': 0, 'Yes': 1}).astype(int)
    X_train = df.loc[:,features].values
    y_train = df.loc[:,['RainTomorrow']].values
    return df, X_train, y_train, features

# 時間窓を設定して平均値、最小値、最大値、標準偏差を取得する
# 時間窓はdatetime型の値を格納したタプルで表す
def time_window(df, target_feature, twin):
    df1 = df[(df['Date'] >= twin[0]) & (df['Date'] <= twin[1])]
    mean_win = df1[target_feature].mean()
    min_win = df1[target_feature].min()
    max_win = df1[target_feature].max()
    std_win = np.std( df1[target_feature] )
    return df1, mean_win, min_win, max_win, std_win

def main():
    prepare()
    df, X_train, y_train, features = preprocess()
    # 時間窓の設定
    win1 = (datetime(2009,1,1), datetime(2009,7,31))
    win2 = (datetime(2017,1,1), datetime(2017,7,31))
    print('\n')
    # 時間窓ごとに、平均値、最小値、最大値を計算
    for win in [win1, win2]:
        print('******* {:^14} ~ {:^14} *******'.format(
            win[0].strftime('%Y-%m-%d'),win[1].strftime('%Y-%m-%d')))
        print('{:^10}\t{:>6}\t{:>6}\t{:>6}\t{:>6}'.format(
                     'Feature', 'Avg', 'Min', 'Max', 'Std'))
        for target_feature in ['Rainfall', 'MaxTemp', 'MinTemp']:
            df1, mean_win1, min_win1, max_win1, std_win1 = time_window(df, target_feature, win)
            print('{:^10}\t{:>6.2f}\t{:>6.2f}\t{:>6.2f}\t{:>6.2f}'.format(
                    target_feature, mean_win1, min_win1, max_win1, std_win1))

if __name__ == '__main__':
    main()

