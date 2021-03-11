from natto import MeCab
freq = {}
# 表層形と品詞を抽出
with MeCab('-F%m,%f[0]') as mc: 
    with open("neko.txt", "r") as f:
        for s in f:
            for w in mc.parse(s, as_nodes=True): 
                # 表層形と品詞をリストに格納
                flist = w.feature.split(',');
                if len(flist)>1:
                    word = flist[0]
                    pos = flist[1]
                    if pos == '名詞': 
                        # 単語のカウントを1増やす
                        if word in freq: 
                            freq[word]=freq[word]+1
                        else:
                            freq[word]=1
 
# 結果を頻度順に表示
sa = sorted(freq.items(), key=lambda x: x[1], reverse=True)
sa[:10]
