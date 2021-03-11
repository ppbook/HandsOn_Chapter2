from natto import MeCab
freq = {}
# 文の単語を空白で区切って出力するオプション
with MeCab('-Owakati') as mc: 
    with open("neko.txt", "r") as f:
        for s in f:
            wordlist = mc.parse(s).split()
            for i in range(len(wordlist)-1):
                # 隣り合う2単語を連結
                w1 = wordlist[i]
                w2 = wordlist[i+1]
                bigram = w1+"-"+w2
                if bigram in freq: 
                    freq[bigram]=freq[bigram]+1
                else:
                    freq[bigram]=1

# 結果を頻度順に表示
sa = sorted(freq.items(), key=lambda x: x[1], reverse=True)
for a in sa:
    if a[1]>=100:
        print(a[0],"(頻度=",a[1],")")
