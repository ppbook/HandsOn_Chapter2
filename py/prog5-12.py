from natto import MeCab
freq = {}
with open("neko.txt", "r") as f:
    for s in f:
        # SentencePiece‚Å•¶‚ð•ªŠ„
        wordlist = sp.EncodeAsPieces(s) 
        for word in wordlist:
            if word in freq:
                freq[word]=freq[word]+1
            else:
                freq[word]=1

# Œ‹‰Ê‚ð•p“x‡‚É•\Ž¦
sa = sorted(freq.items(), key=lambda x: x[1], reverse=True)
for a in sa:
    if a[1]>=10:
        if len(a[0])>=5:
            print(a[0],"(•p“x=",a[1],")")
