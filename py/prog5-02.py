import requests, io

# Shift_JISでファイルを開き、最初の30行を表示
with open("wagahaiwa_nekodearu.txt", "r", 
          encoding='Shift_JIS') as f:
    lines = f.readlines()
    for s in lines[0:30]:
      print(s, end='')
