from google.colab import drive 
drive.mount('/content/drive')
%cd /content/drive/MyDrive

# word2vecのベクトル読み込みのためのモジュールをインポート
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# 学習済みベクトルを読み込む
model = KeyedVectors.load_word2vec_format(\
    'jawiki.word_vectors.100d.txt', binary=False) 
