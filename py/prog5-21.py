# word2vecで読み込んだモデルを、2次元のnumpy配列に変換
# word_indexは、単語をキーとしてベクトルを返す連想配列
import numpy as np
from tensorflow import keras
word_index={}
for key in model.vocab:
    word_index[key]=len(word_index)

num_words = len(word_index)+2 # 語彙の数
w2v_size = 100 # ベクトルの次元数

word_matrix = np.zeros((num_words, 100), dtype="float32")
# ID=0が[PAD]、ID=1が[UNK]に対応。どちらも0ベクトル。
word_matrix[0] = np.zeros((100), dtype="float32")
word_matrix[1] = np.zeros((100), dtype="float32")
for key in word_index:
    # 読み込んだ辞書の最初の単語が、ID=2
    word_matrix[2+word_index[key]]=model[key]

# テキスト集合をの読み込み
docs2 = []
for s in docs: 
    sl = s.split()
    sl2 = []
    for w in sl:
        if w in word_index:
            sl2.append(2+word_index[w]) 
        else:
            sl2.append(1) # OOV単語
    docs2.append(sl2)

# np.arrayで行列（2次元配列）を生成するために、
# 各文（単語リスト）の長さをそろえる。
# デフォルトでは、文の最後に0が追加される。
docs3 = keras.preprocessing.sequence.pad_sequences(\
    docs2, value=0, padding="post", maxlen=100)
docs4 = np.array(docs3) 
labels2 = np.array(labels)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y, train_i, test_i = \
    train_test_split(docs4, labels2, inds, test_size=0.2) 
