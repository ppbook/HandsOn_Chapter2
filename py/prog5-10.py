from natto import MeCab
freq = {}
# 表層形、品詞細分類(1)を抽出
with MeCab('-F%m,%f[1]') as mc: 
    with open("neko.txt", "r") as f:
        for s in f:
            for w in mc.parse(s, as_nodes=True):
                # 表層形、品詞細分類(1)をリストに格納
                flist = w.feature.split(',');
                if len(flist)>1:
                    word = flist[0]
                    subpos = flist[1]
                    # 品詞細分類(1)が固有名詞なら、
                    # 頻度をカウント
                    if subpos == '固有名詞': 
                        word = flist[0]
                        if word in freq:
                            freq[word]=freq[word]+1
                        else:
                            freq[word]=1

# 結果を頻度順に表示
sa = sorted(freq.items(), key=lambda x: x[1], reverse=True)
sa[:10]
