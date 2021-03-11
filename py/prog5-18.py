import numpy as np

# scikitlearnから、教師あり学習に必要なモジュールをインポート
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# pythonのリストを、numpy配列に変換
X = np.array(docs2) 
y = np.array(labels)

# 訓練データとテストデータに分割
train_x, test_x, train_y, test_y, train_i, test_i \
    = train_test_split(docs2, labels, inds, \
    test_size=0.2)

# scikitlearnから、SVMのモジュールをインポート	
from sklearn.svm import SVC

# SVMを学習
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(train_x, train_y) 
