# LdaMalletのインポート
from gensim.models.wrappers import LdaMallet

# LDAの学習
model = LdaMallet("Mallet/bin/mallet",
    corpus=corpus, num_topics=100, 
    id2word=dictionary, alpha=3.0, iterations=1000) 

# 結果の表示
model.print_topics(100, num_words=5)
