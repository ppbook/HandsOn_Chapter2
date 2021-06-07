# プログラム 2.1

import pandas as pd
import numpy as np
# SVMのクラスをインポート
from sklearn.svm import SVC
# ナイーブベイズ分類器のクラスをインポート
from sklearn.naive_bayes import GaussianNB, MultinomialNB
# ランダムフォレスト、勾配ブースティングのクラスをインポート
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import pandas as pd
# One-hotエンコーディングのクラスをインポート
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# データ準備と前処理
def prepare():
    #!kaggle datasets download -d kendallgillies/video-game-sales-and-ratings
    get_ipython().system('unzip video-game-sales-and-ratings.zip')
    # ビデオゲームの評価データを使用
    # 分類に使用する特徴量
    df_train = pd.read_csv('Video_Game_Sales_as_of_Jan_2017.csv')
    # プラットフォーム、ジャンルなどのカテゴリ特徴量
    features = ['Platform', 'Year_of_Release', 'Genre', 'Publisher']
    print(set(df_train['Rating'].values))
    # 欠損値を除去
    df_train.dropna(how='any', inplace=True)
    # データ数が少ない評価のものは対象外とする
    drop_idx = df_train.index[df_train['Rating'].isin(['K-A', 'RP', 'AO', 'EC'])]
    # 条件にマッチした行を削除
    df_train.drop(drop_idx, inplace=True)
    X_train = df_train.loc[:,features].values
    y_train = df_train.loc[:,['Rating']].values
    print(df_train)
    return X_train, y_train, features

# One-hotエンコーディング
def makeOneHotVector(features, X_train):
    enc = OneHotEncoder(categories='auto', 
        sparse=False, dtype=np.float32)
    df = pd.DataFrame(X_train, columns=features)
    num = len(df)
    newX, data = [], {}
    for f in features:
        data[f] = enc.fit_transform(df.loc[:,[f]])
    # 作成した列ごとのOne-hotベクトルを横に連結する
    newX = np.array([])
    for f in features:
        if len(newX) == 0:
            newX = data[f]
            continue
        newX = np.concatenate((newX, data[f]), axis=-1)
        print('DIM:', len(data[f][0]) )
    return newX

# 機械学習手法による学習と予測
# 引数clfに指定した機械学習手法を用いる
def train_eval(data, clf):
    X_train, y_train, X_test, y_test = data
    print('-------- {} ---------'.format(clf.__class__.__name__))
    clf.fit(X_train, y_train.ravel())
    print( 'Accuracy: %.2lf' % clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=1))

def main():
    X_train, y_train, features = prepare()
    # One-hotエンコーディングを用いてカテゴリ変数をベクトル化
    X_train = makeOneHotVector(features, X_train)
    print(len(X_train[0]))
    X_train, X_test, y_train, y_test =              train_test_split(X_train, y_train, random_state=0, train_size=0.9)
    data = [X_train, y_train, X_test, y_test]
    # ランダムフォレスト、勾配ブースティング、
    # ナイーブベイズ分類器(Gaussian, Multinomial)、
    # SVMで学習・分類
    rf = RandomForestClassifier(max_depth=5, random_state=0)
    gbc = GradientBoostingClassifier()
    gnb = GaussianNB()
    mnb = MultinomialNB()
    svm = SVC(C=1.0)
    clfs = [rf, gbc, gnb, mnb, svm]
    for clf in clfs:
        train_eval(data, clf)

if __name__ == '__main__':
    main()

