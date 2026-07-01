import json, os, time
from openai import OpenAI

INPUT_JSON = "vocab.json"
INPUT_WORDS = "new_words.txt"
OUTPUT_JSON = "vocab.json"
BATCH_SIZE = 25
NEW_CLUSTER = 32
MODEL = "deepseek-v4-flash"

SYSTEM = (
    "You are a lexicographer building GRE-style flashcards. For each word, return its "
    "part of speech (abbreviated: adj., noun, verb, adv.), a concise one-sentence "
    "definition, and one natural example sentence using the word. Respond ONLY with JSON: "
    '{"results":[{"word":"...","pos":"...","def":"...","sentence":"..."}]}. '
    "Preserve the exact input spelling and order."
)

def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}

def save(path, data):
    json.dump(data, open(path, "w"), ensure_ascii=False, indent=2)

def load_new_words(path):
    seen = set()
    words = []
    with open(path) as f:
        for line in f:
            word = line.strip()
            key = word.lower()
            if not word or key in seen:
                continue
            seen.add(key)
            words.append(word)
    return words

def get_json_results(response):
    content = response.choices[0].message.content
    data = json.loads(content)
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError("DeepSeek response JSON did not contain a results list")
    return results

def add_batch(client, vocab, batch):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Words:\n" + "\n".join(batch)},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    items = get_json_results(response)
    by_word = {
        str(item.get("word", "")).strip().lower(): item
        for item in items
        if isinstance(item, dict)
    }

    missing = [word for word in batch if word.lower() not in by_word]
    if missing:
        raise ValueError(f"missing model results for: {', '.join(missing)}")

    additions = {}
    incomplete = []
    for word in batch:
        item = by_word.get(word.lower())
        if not all(str(item.get(field, "")).strip() for field in ("pos", "def", "sentence")):
            incomplete.append(word)

        additions[word] = {
            "cluster": NEW_CLUSTER,
            "pos": str(item.get("pos", "")).strip(),
            "def": str(item.get("def", "")).strip(),
            "sentence": str(item.get("sentence", "")).strip(),
        }

    if incomplete:
        raise ValueError(f"incomplete model results for: {', '.join(incomplete)}")

    vocab.update(additions)

def main():
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    vocab = load(INPUT_JSON)
    words = load_new_words(INPUT_WORDS)
    existing = {word.lower() for word in vocab}
    pending = [word for word in words if word.lower() not in existing]

    print(f"{len(words)} words in {INPUT_WORDS}")
    print(f"{len(pending)} genuinely new words")

    total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i:i + BATCH_SIZE]
        batch_number = i // BATCH_SIZE + 1

        for attempt in range(3):
            try:
                add_batch(client, vocab, batch)
                save(OUTPUT_JSON, vocab)
                print(
                    f"batch {batch_number}/{total_batches}: +{len(batch)} "
                    f"(total {len(vocab)})"
                )
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                if attempt == 2:
                    raise
                print(f"  batch {batch_number} failed ({e}); retry in {wait}s")
                time.sleep(wait)

        time.sleep(1)

    print(f"done - {len(vocab)} words in {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
