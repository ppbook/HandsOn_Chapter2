import io
# 正規表現用モジュールをインポート
import re
#(元のファイル名,処理後のファイル名)のリスト。
textlist = [
('wagahaiwa_nekodearu','neko'),
    ('kaijin_nijumenso','kaijin'),
]
for i in range(len(textlist)):
    text = textlist[i][0]
    text2 = textlist[i][1]
  
    # 各前処理の結果を一時保存するリスト
    tmplist = []
    tmplist2 = []
    tmplist3 = []

    # ファイル冒頭のメタデータ削除
    # nはこれまでに読んだ区切り行の数
    n=0
    with open(text+".txt", "r", encoding='Shift_JIS') as f:
        for s in f:
            if n>=2: 
                tmplist.append(s)
            if s.find("--------")>=0:
                n=n+1

    # ファイル末尾のメタデータ削除
    # nは連続して読んだ空行の数
    n=0
    for s in tmplist:
        #改行コードが1文字分
        if len(s)==1:
            n=n+1
            if n>=3:
              break
        else:
            n=0
        tmplist2.append(s)
    
    # 本文中の注釈を削除
    for s in tmplist2:
        s2 = re.sub(r'《.*?》', "", s) 
        s3 = re.sub(r'［＃.*?］', "", s2)
        s4 = re.sub(r'｜',"",s3)
        tmplist3.append(s4)

    # 各行を、文に分割し、Unicodeでファイルに出力
    with open(text2+".txt", "w", encoding='utf-8') as f:
        for line in tmplist3:
            sentlist = line.split('。')
            for i in range(len(sentlist)):
                s = sentlist[i] 
                s = re.sub(r'\n','',s)
                if len(s)>0:
                    f.write(s)
                    if i<len(sentlist)-1:
                        f.write('。')
                    f.write('\n')
