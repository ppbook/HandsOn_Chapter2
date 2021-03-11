from natto import MeCab
corpus = []
labels = []
with MeCab('-Owakati') as mc: 
  with open("neko.txt", "r") as f:
    for s in f:
      s2 = mc.parse(s)#単語をスペースで区切る　
      corpus.append(s2.split())

# gensimから、word2vec用モジュールをインポート
from gensim.models import word2vec

# corpusに格納されたデータから、分散表現ベクトルを学習
model = word2vec.Word2Vec(sentences=corpus, \
sg=1, min_count=3,window=5) 
