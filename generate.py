import os, json, time
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

INPUT_XLSX = "clustered_wordsv2.xlsx"
SHEET = "sheet2"
OUTPUT_JSON = "vocab.json"
BATCH_SIZE = 25

SYSTEM = (
    "You are a lexicographer building GRE-style flashcards. For each word, return its "
    "part of speech (abbreviated: adj., noun, verb, adv.), a concise one-sentence "
    "definition, and one natural example sentence using the word. Respond ONLY with JSON: "
    '{"results":[{"word":"...","pos":"...","def":"...","sentence":"..."}]}. '
    "Preserve the exact input spelling and order."
)

def load_done():
    return json.load(open(OUTPUT_JSON)) if os.path.exists(OUTPUT_JSON) else {}

def save(data):
    json.dump(data, open(OUTPUT_JSON, "w"), ensure_ascii=False, indent=2)

def main():
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET).dropna(subset=["Word", "Cluster"])
    rows = [{"word": str(r.Word).strip(), "cluster": int(r.Cluster)} for r in df.itertuples()]

    done = load_done()
    pending = [r for r in rows if r["word"] not in done]
    print(f"{len(pending)} words left of {len(rows)} total")

    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i:i + BATCH_SIZE]
        words = [r["word"] for r in batch]
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": "Words:\n" + "\n".join(words)}],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                items = json.loads(resp.choices[0].message.content)["results"]
                by_word = {it["word"].strip().lower(): it for it in items}
                for r in batch:
                    it = by_word.get(r["word"].lower(), {})
                    done[r["word"]] = {
                        "cluster": r["cluster"],
                        "pos": it.get("pos", ""),
                        "def": it.get("def", ""),
                        "sentence": it.get("sentence", ""),
                    }
                save(done)
                print(f"batch {i//BATCH_SIZE+1}: +{len(batch)} (total {len(done)})")
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  batch failed ({e}); retry in {wait}s")
                time.sleep(wait)
        time.sleep(1)

    print(f"done — {len(done)} words in {OUTPUT_JSON}")

if __name__ == "__main__":
    main()