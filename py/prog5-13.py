from natto import MeCab
freq = {}
stopwords ={'ÇªÇÍ','Ç±ÇÍ','éÑ','ÇÃ','Ç‡ÇÃ','ÇÊÇ§','ÇÒ',
'ÇÁ','ÇΩÇﬂ','ÇªÇ§'}
with MeCab('-F%m,%f[0]') as mc:
    with open("neko.txt", "r") as f:
        for s in f:
            for w in mc.parse(s, as_nodes=True):
                flist = w.feature.split(',');
                if len(flist)>1:
                    word = flist[0]
                    pos = flist[1]
                    if pos == 'ñºéå': 
                        if not word in stopwords: 
                            if word in freq:
                                freq[word]=freq[word]+1
                            else:
                                freq[word]=1

#íPåÍIDé´èëÇç\íz
word2id = {}
id2word = []
for s in freq:
    if not s in word2id:
        if freq[s]>=10:
            word2id[s]=len(id2word)
            id2word.append(s)

print('Å°IDèá')
for i in range(min(10,len(id2word))):
    print('ID=',i,':',id2word[i])

print('')
print('Å°ïpìxèá')
sa = sorted(freq.items(), key=lambda x: x[1], reverse=True)
for a in sa[:10]:
    print('ID='+str(word2id[a[0]]),":",a[0],
    "(ïpìx=",a[1],")")
