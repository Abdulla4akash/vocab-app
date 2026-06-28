import json, csv

data = json.load(open("vocab.json"))
rows = sorted(data.items(), key=lambda kv: (kv[1]["cluster"], kv[0]))

with open("anki_import.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    for word, d in rows:
        front = word
        back = f'<b>{d["pos"]}</b><br>{d["def"]}<br><br><i>{d["sentence"]}</i>'
        tag = f'cluster::{d["cluster"]:02d}'
        w.writerow([front, back, tag])

print(f"wrote anki_import.csv with {len(rows)} cards")