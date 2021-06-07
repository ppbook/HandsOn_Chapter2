# プログラム2.2

get_ipython().system('pip3 install category_encoders')
import pandas as pd
# category_encodersをインポート
import category_encoders as cate_enc
# ニューラルネットワークのクラスをインポート
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# データの準備
def prepare():
    #!kaggle datasets download -d cms/hospital-general-information
    get_ipython().system('unzip hospital-general-information.zip')

def preprocess():
    df = pd.read_csv('HospInfo.csv')
    print(df)
    print(df)
    # 病院のデータ
    features = ['City', 'State',
                'County Name', 'Hospital Type',
                'Emergency Services', 
                'Meets criteria for meaningful use of EHRs',
                'Mortality national comparison', 
                'Safety of care national comparison',
                'Readmission national comparison', 
                'Patient experience national comparison',
                'Effectiveness of care national comparison', 
                'Timeliness of care national comparison',
                'Efficient use of medical imaging national comparison']
    ignores = []
    for f in df.columns.values:
        if not f in features: 
            ignores.append(f)
    ignores.remove('Hospital overall rating')
    ratings = ['1', '2', '3', '4', '5']
    mp = {'1':0, '2':1, '3':2, '4':3, '5':4}
    df = df[df['Hospital overall rating'].isin(ratings)]
    df['Hospital overall rating'].replace(mp, inplace=True)
    df.drop(ignores, axis=1, inplace=True)
    # One-hotエンコーディング
    ohe = cate_enc.OneHotEncoder(cols=features, handle_unknown='impute')
    ndf = ohe.fit_transform(df)
    # 病院の評価を予測対象とする
    y = ndf.loc[:,['Hospital overall rating']].values.ravel()
    ndf.drop(columns=['Hospital overall rating'], inplace=True)
    return ndf, y, ratings

def main():
    prepare()
    ndf, y, ratings = preprocess()
    # One-hotエンコーディング結果の確認
    print(ndf.loc[:,ndf.columns.values[:5]].head())
    print(ndf.loc[:,ndf.columns.values[3565:3570]].head())
    # テストデータと学習データに分割して
    # ニューラルネットワークによるratingの学習と予測
    X = ndf.loc[:,ndf.columns.values].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, train_size=0.7)
    clf = MLPClassifier(solver='adam', alpha=1e-5, 
                        hidden_layer_sizes=(100,), 
                        activation='tanh',
                        random_state=1, max_iter=3000)
    clf.fit(X_tr, y_tr)
    y_pre = clf.predict(X_te)
    print(classification_report(y_te, y_pre, target_names=ratings, zero_division=1))

if __name__ == '__main__':
    main()

