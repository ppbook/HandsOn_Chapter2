from natto import MeCab
import math
import numpy as np

# gensimをインポート
import gensim
# 前処理に必要なモジュールを、gensimからインポート
from gensim import corpora, models, similarities

docs=[]
stopwords ={'それ','これ','私','の','もの','よう','ん','ら','ため','そう'}

with MeCab('-F%m,%f[0]') as mc:
    with open("neko.txt", "r") as f:
        for s in f:
            # 各文を単語リストに変換し、リストdocsに加える
            wordlist = []
            for w in mc.parse(s, as_nodes=True): 
                flist = w.feature.split(',');
                if len(flist)>1: 
                    word = flist[0] 
                    pos = flist[1]
                    if pos == '名詞':
                        if not word in stopwords:
                            wordlist.append(word)
            docs.append(wordlist)

# 辞書を構築
dictionary = corpora.Dictionary(docs) 
#dictionary.filter_extremes(no_below=2)
# 構築された辞書を使い、単語リストをbag-of-wordsに変換
corpus = [dictionary.doc2bow(doc) for doc in docs] 
