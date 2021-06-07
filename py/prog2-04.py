# プログラム2.4

import pandas as pd
import numpy as np
import collections as colle
from sklearn.preprocessing import LabelEncoder
# SVMを用いるためにインポート
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report as clf_report
from sklearn.impute import SimpleImputer
# 特徴量ハッシングのためにFeatureHasherクラスをインポート
from sklearn.feature_extraction import FeatureHasher

#!kaggle datasets download -d mariotormo/complete-pokemon-dataset-updated-090420
get_ipython().system('unzip complete-pokemon-dataset-updated-090420.zip')

# データの準備
def prepare(le_flag=False):
    df = pd.read_csv('pokedex_(Update_05.20).csv')
    # 欠損値を文字列'NULL'に置換
    for f in df.columns.values:
        df[f].fillna('NULL', inplace=True)

    # 予測モデルを学習する際に用いる特徴量
    features = ['name', 'german_name', 'japanese_name', 
                'status', 'species', 
                'type_1', 'type_2', 
                'ability_1', 'ability_2', 
                'ability_hidden', 
                'egg_type_1', 'egg_type_2']
    df = pd.DataFrame(df, columns=features)
    nf = []
    # 特徴量をラベルエンコーディング
    if le_flag:
        for u in df.columns.values:
            if u != 'status':
                le = LabelEncoder()
                enc = le.fit_transform(df[u].values)
                df[u] = enc
    # n_features 個の特徴量を用いる
    # 'status'は予測クラスのラベルとして使う
    for f in features:
        if f != 'status':
            nf.append(f)
    features = nf
    df['status'].replace({'Legendary':1, 'Sub Legendary':1, 'Mythical':1, 'Normal':0}, 
                         inplace=True)
    X_train = df.loc[:,features].values
    y_train = df.loc[:,['status']].values.ravel()
    
    # 伝説のポケモンが少ないため、クラスのバランスを調整する
    positive_cnt = np.sum(y_train)
    negative_cnt = 0
    cnt = colle.defaultdict(int)
    X_tra = []
    y_tra = []
    for i in range(len(X_train)):
        if y_train[i] == 0:
            if negative_cnt == positive_cnt:
                continue
            negative_cnt += 1
        
        X_tra.append(X_train[i])
        y_tra.append(y_train[i])
    print('Num of Features: {}'.format(len(features)))
    return X_tra, y_tra, features

# 欠損値の補完を行い、テストデータと学習データに分割
def preprocess(X_train, y_train, features):
    # データの種類を数える
    dl = []
    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            dl.append(X_train[i][j])
    cnt = colle.Counter(dl)
    n_features = len(cnt)
    X_train, X_test, y_train, y_test = train_test_split(
                         X_train, y_train, random_state=3, 
                         train_size=0.5, stratify=y_train)
    X_train = pd.DataFrame( X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)
    return X_train, y_train, X_test, y_test, n_features

# 特徴量ハッシング(特徴量の種類を減らす)
def featureHashing(X_train, X_test, n_features, features):
    X_train_dict = pd.DataFrame(X_train, columns=features)
    X_test_dict = pd.DataFrame(X_test, columns=features)
    fh = FeatureHasher(
           n_features=n_features, input_type='string')
    fh_train, fh_test = [], []
    for f in features:
        if len(fh_train) == 0:
            fh_train = fh.transform(X_train_dict[f]).toarray()
            fh_test = fh.transform(X_test_dict[f]).toarray()
        else:
            fh_train = fh_train + fh.transform(X_train_dict[f]).toarray()
            fh_test = fh_test + fh.transform(X_test_dict[f]).toarray()
    return fh_train, fh_test

def main():
    tnames = ['Normal', 'Legend']
    X_train, y_train, features = prepare(le_flag=False)
    X_train, y_train, X_test, y_test, original_n_features =                                       preprocess(X_train, y_train, features)
    print('original dimension: %d' % len(X_train.columns))
    print('original kinds of features: %d'                              % original_n_features)
    # 特徴量ハッシングにより特徴量の種類を
    # 全体でn_features種類に減らす
    n_features = 6
    fhtrain, fhtest = featureHashing(X_train, X_test, n_features, features)
    df = pd.DataFrame(fhtrain, columns=list(range(n_features)))
    df.to_csv('./feature_hashing.csv')
    # カテゴリ特徴量をもとに、ポケモンが伝説の
    # ポケモンか否かを判別するモデルをSVMにより学習する
    svc = SVC()
    svc.fit(fhtrain, y_train)
    print('Accuracy(With Feature Hashing):\t%.3lf' % (
            svc.score(fhtest, y_test)))
    y_pred = svc.predict(fhtest)

    print(classification_report(y_test, y_pred, target_names=tnames))

    # 特徴量ハッシングしない場合
    # ラベルエンコーディングしてから学習
    X_train, y_train, features = prepare(le_flag=True)
    X_train, y_train, X_test, y_test, original_n_features =                                       preprocess(X_train, y_train, features)
    svc = SVC()
    svc.fit(X_train, y_train)
    print('Accuracy(Without Feature Hashing):\t%.3lf' % (svc.score(X_test, y_test)))
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred,target_names=tnames))

if __name__ == '__main__':
    main()
    

