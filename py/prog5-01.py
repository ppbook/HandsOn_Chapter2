# ダウンロード、ファイル書き込み用モジュールのインポート
import requests, io

# 必要なURLのリストを準備する
urllist = [
"000148/files/789_ruby_5639.zip",#吾輩は猫である
"001779/files/57228_ruby_58697.zip"#怪人二十面相
] 

#URLのリストをもとに、ファイルをダウンロードする
for i in range(len(urllist)):
    url = "https://www.aozora.gr.jp/cards/"+urllist[i]
    fn = "text-"+str(i)
    res = requests.get(url, stream=True) 
    with open(fn+".zip", "wb") as handle:
        for chunk in res.iter_content(chunk_size=512):
          if chunk:
              handle.write(chunk)

# ダウンロードしたファイルを解凍する
!unzip "*.zip"
