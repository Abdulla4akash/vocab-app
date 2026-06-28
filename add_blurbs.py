import os, json, time
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

INPUT_JSON = "vocab.json"
OUTPUT_JSON = "vocab_blurbs.json"
BATCH_SIZE = 15

SYSTEM = (
    "You write vivid, memorable word-stories like Vocabulary.com — the kind that make a word "
    "stick by giving it a hook the reader won't forget. For each word, write a 'blurb': 2-3 "
    "sentences. Lead with a concrete image, a scenario, the word's roots, or how it sounds — "
    "anything that anchors the meaning in something memorable. Never just restate the definition. "
    "Be specific, a little playful, and confident.\n\n"
    "Here is the quality bar:\n\n"
    "WORD: ephemeral (lasting a very short time)\n"
    "BLURB: \"Mayflies live for a single day, and the word ephemeral comes from the Greek for "
    "'lasting a day' — ephemeros. Anything ephemeral is here and then gone: a snowflake on a warm "
    "palm, a Snapchat, the perfect mood before someone ruins it.\"\n\n"
    "WORD: obfuscate (to deliberately make something unclear)\n"
    "BLURB: \"Buried inside obfuscate is the Latin for 'to darken' — fuscus, dusky. When politicians "
    "obfuscate, they don't lie outright; they pour fog over the truth until you give up trying to "
    "see it. Think of a squid clouding the water to escape.\"\n\n"
    "Match that style. Respond ONLY with JSON: "
    '{"results":[{"word":"...","blurb":"..."}]}. Preserve exact input spelling and order.'
)

def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}

def save(path, data):
    json.dump(data, open(path, "w"), ensure_ascii=False, indent=2)

def main():
    vocab = load(INPUT_JSON)
    done = load(OUTPUT_JSON)

    pending = [w for w in vocab if w not in done]
    # For a test run, uncomment the next line to only do the first 10:
    # pending = pending[:10]
    print(f"{len(pending)} blurbs left of {len(vocab)} total")

    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i:i + BATCH_SIZE]
        # give the model the word + its definition for context
        lines = [f'{w} — {vocab[w]["def"]}' for w in batch]
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": "Words:\n" + "\n".join(lines)}],
                    response_format={"type": "json_object"},
                    temperature=0.8,
                )
                items = json.loads(resp.choices[0].message.content)["results"]
                by_word = {it["word"].strip().lower(): it for it in items}
                for w in batch:
                    it = by_word.get(w.lower(), {})
                    done[w] = it.get("blurb", "")
                save(OUTPUT_JSON, done)
                print(f"batch {i//BATCH_SIZE+1}: +{len(batch)} (total {len(done)})")
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  batch failed ({e}); retry in {wait}s")
                time.sleep(wait)
        time.sleep(1)

    print(f"done — {len(done)} blurbs in {OUTPUT_JSON}")

if __name__ == "__main__":
    main()