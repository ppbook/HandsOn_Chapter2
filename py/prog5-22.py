# numpy, tensorflow, kerasをインポート
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding
# 空のニューラルネットワークを生成
mycnn = keras.Sequential()
# Embedding層を追加
mycnn.add(Embedding(num_words, w2v_size, \
    weights=[word_matrix], input_length=100, \
    mask_zero=True, trainable=False)) 
# 畳み込み層を追加
mycnn.add(keras.layers.Conv1D(100,1, activation=tf.nn.relu))
# プーリング層を追加
mycnn.add(keras.layers.GlobalMaxPooling1D())
# 全結合層を追加
mycnn.add(keras.layers.Dense(1, activation=tf.nn.sigmoid)) 
# 学習のためのモデルを設定
mycnn.compile(optimizer=tf.optimizers.Adam(),
    loss="binary_crossentropy", metrics=["accuracy"])
# 構築結果を表示
mycnn.summary()
