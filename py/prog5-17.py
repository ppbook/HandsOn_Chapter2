from natto import MeCab
# scikitlearnから、tf-idfベクトル用モジュールをインポート
from sklearn.feature_extraction.text import TfidfVectorizer

docs = []
labels = []
inds = []
with MeCab('-Owakati') as mc:
    with open("neko.txt", "r") as f:
        for s in f:
            s2 = mc.parse(s)#単語をスペースで区切る
            docs.append(s2)
            labels.append(1) 
            inds.append(len(inds)) 
    with open("kaijin.txt", "r") as f:
        for s in f:
            s2 = mc.parse(s)#単語をスペースで区切る
            docs.append(s2)
            labels.append(0)
            inds.append(len(inds))

vectorizer = TfidfVectorizer(use_idf=True) 
vecs = vectorizer.fit_transform(docs)

docs2 = vecs.toarray()
