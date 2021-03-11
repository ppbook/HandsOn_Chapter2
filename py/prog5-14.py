from natto import MeCab
import math
import numpy as np
freq = {}
dfreq = {}
# docsに、(文, 単語リスト)のペアを格納
docs = []
with MeCab('-F%m,%f[0]') as mc:
    with open("neko.txt", "r") as f:
        for s in f:
            wordlist = []
            # 文中に出た単語の集合
            wordset = set()
            for w in mc.parse(s, as_nodes=True):
                flist = w.feature.split(',');
                if len(flist)>1:
                    word = flist[0]
                    pos = flist[1]
                    if pos == '名詞':
                        wordset.add(word)
                        wordlist.append(word)
                        if word in freq:
                            freq[word]=freq[word]+1
                        else:
                            freq[word]=1
            docs.append((s,wordlist))
            # wordset中の各単語のカウントを1増やす
            for w in wordset: 
                if w in dfreq:
                    dfreq[w]=dfreq[w]+1
                else:
                    dfreq[w]=1


#単語ID辞書を構築
word2id = {}
for s in freq:
    if not s in word2id:
        if freq[s]>=10:
            word2id[s]=len(word2id)

#次元数
dim = len(word2id)
docnum = len(docs)

# docs2に、（文, 文ベクトル）のペアを格納
docs2 = []
for doc in docs:
    vec = np.zeros(dim)
    wordlist = doc[1]
    wordset = {}
    for w in wordlist:
        if w in word2id: 
            id = word2id[w] 
            df = dfreq[w]
            # IDF値の計算
            idf = math.log(docnum/df)
            # tf-idfベクトルを作る。
            # +idfを+1に変えると、bag-of-wordsになる
            vec[id]=vec[id]+idf
    # ベクトルを、長さ1になるように正規化
    norm = np.linalg.norm(vec, ord=2) 
    if norm>0: 
        vec = vec/norm
    docs2.append((doc[0], vec))

# 本文中の一文を、クエリ文として設定。
query = docs2[6]
print("クエリ文："+query[0])
print()
firstvec = query[1]

simlist = {}
# 本文中の各文と、クエリ文の類似度を、simlistに登録
for id in range(len(docs2)): 
    doc = docs2[id]
    vec = doc[1]
    sim = np.inner(vec, firstvec)
    simlist[id] = sim

# 類似度順に文をソート
sa = sorted(simlist.items(), key=lambda x: x[1], reverse=True)

# 類似度の上位10件を表示
for i in range(10):
    ida = sa[i]
    id = ida[0]
    doc = docs2[id]
    print("類似度："+str(ida[1]))
    print("類似文："+doc[0])

