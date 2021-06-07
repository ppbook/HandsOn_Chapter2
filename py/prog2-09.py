# プログラム2.9

import sys
import numpy as np
import pandas as pd
# 可視化用にインポート
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# データの準備
def prepare():
    #!kaggle datasets download -d shenba/time-series-datasets
    get_ipython().system('unzip time-series-datasets.zip ')
    data = pd.read_csv('daily-minimum-temperatures-in-me.csv')
    print(len(data))
    # 欠損値（異常値には'?'が含まれる）を除外
    data = data[~data['Daily minimum temperatures'].str.contains('\?')]
    print(len(data))
    features = []
    for f in data.columns.values:
        if f != 'Date':
            features.append(f)
    # 日付の文字列をdatetime型に変換する
    data['Date'] = pd.to_datetime(data['Date'])
    return data, features

def main():
    df, features = prepare()
    print(len(df))
    s = df['Daily minimum temperatures']
    # インデックスに日付の項目を指定
    s.index = df['Date']
    # 1か月，1年, 5年単位での平均気温を表示
    # Windowの中心に平均値を格納( center=True)
    for span in [30, 365, 1825]:
        print( s.rolling( span, center=True ).mean() )
        rol = s.rolling( span, center=True ).mean()
        # 計算結果の入っていない行を削除
        rol = rol.dropna()
        plt.figure(figsize=(8, 4))
        plt.plot(rol)
        plt.savefig('{}.png'.format(span), dpi=500)
        # 算出した結果をCSVファイルに出力
        rol.to_csv('./avg_%d.txt' % span)

if __name__ == '__main__':
    main()

