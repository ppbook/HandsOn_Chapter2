# プログラム2.3

# ラベルエンコーディング用にインポート
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# 頻度エンコーディングと順位エンコーディング用に
# ライブラリcollectionsをインポート
import collections as colle

# データの準備
def prepare():
    #!kaggle datasets download -d shabab477/bangladesh-food-safety-authority-restaurant-rating
    get_ipython().system('unzip bangladesh-food-safety-authority-restaurant-rating.zip')
    # バングラデシュ食品安全局のレストランの
    # 評価のデータを使用
    # 分類に使用する特徴量
    features = ['area', 'city', 'food_type_name']
    df_train = pd.read_csv('bd-food-rating.csv')
    # 欠損値を最頻値で穴埋め
    for f in features:
        df_train[f].fillna(df_train[f].mode()) 
    X_train = df_train.loc[:,features].values
    y_train = df_train.loc[:,['bfsa_approve_status']].values
    return X_train, y_train, features

# ラベルエンコーディング
def label_encoding(X_train, features):
    ndf = pd.DataFrame(X_train, columns=features)
    ledf = pd.DataFrame([], columns=features)
    ledic = {}
    for f in features:
        le = LabelEncoder()
        encoded = le.fit_transform(ndf[f].values)
        ledic[f] = le
        ledf[f] = encoded
    return ledic, ledf

# 頻度エンコーディング
def freq_encoding(X_train, features):
    ndf = pd.DataFrame(X_train, columns=features)
    fdf = pd.DataFrame([], columns=features)
    fdic = {}
    for f in features:
        cnt = colle.Counter(ndf[f].values)
        freq_dict = dict(cnt.most_common())
        print(freq_dict) 
        fdic[f] = freq_dict
        encoded = ndf[f].map(lambda x: cnt[x]).values
        fdf[f] = encoded
    return fdic, fdf

# 順位エンコーディング
def ranked_freq_encoding(X_train, features):
    ndf = pd.DataFrame(X_train, columns=features)
    ldf = pd.DataFrame([], columns=features)
    ldic = {}
    for f in features: 
        cnt = colle.Counter(ndf[f].values)
        label_dict = {keyfreq[0]:i for i, keyfreq in enumerate(cnt.most_common(), start=1)}
        ldic[f] = label_dict
        encoded = ndf[f].map(lambda x: label_dict[x]).values
        ldf[f] = encoded
    return ldic, ldf

def main():
    X_train, y_train, features = prepare()
    df = pd.DataFrame(X_train, columns=features)
    print(len(df))
    # ラベルエンコーディング
    print('Label encoding')
    ledic, ledf = label_encoding(X_train, features)
    print(ledf[:5])
    decoded = ledf['area'][:5]
    print(decoded)
    print('------')

    # 頻度エンコーディング
    print('Freq. encoding')
    fdic, fdf = freq_encoding(X_train, features)
    print(fdf[:5])
    # 確認(頻度が衝突するので複数のキーがマッチ)
    for v in fdf['area'][:5]:
        keys = [ky for ky, val in fdic['area'].items() if val == v]
        print('{}:  {}'.format(v, ', '.join(keys)))
    print('------')
    
    # 順位エンコーディング
    print('Ranked frequency encoding')
    ldic, ldf = ranked_freq_encoding(X_train, features)
    print(ldf[:5])
    for v in ldf['area'][:5]:
        keys = [ky for ky, val in ldic['area'].items() if val == v]
        print('{}:  {}'.format(v, ', '.join(keys)))

if __name__ == '__main__':
    main()
    

