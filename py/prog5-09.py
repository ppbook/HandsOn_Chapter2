from natto import MeCab
freq = {}
# 表層形、品詞、原型を抽出
with MeCab('-F%m,%f[0],%f[6]') as mc: 
    with open("neko.txt", "r") as f:
        for s in f:
            for w in mc.parse(s, as_nodes=True):
                # 表層形、品詞、原型をリストに格納
                flist = w.feature.split(',');
                if len(flist)>2:
                    pos = flist[1]
                    root = flist[2]
                    # 品詞が動詞だったら、原型の頻度を1増やす
                    if pos == '動詞': 
                        if root in freq:
                            freq[root]=freq[root]+1
                        else:
                            freq[root]=1

#結果を頻度順に表示
sa = sorted(freq.items(), key=lambda x: x[1], reverse=True)
sa[:10] 
