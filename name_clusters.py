import os, json, time
from collections import defaultdict
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

INPUT_JSON = "vocab.json"
OUTPUT_JSON = "clusters.json"
SAMPLE_SIZE = 90  # words shown to the model per cluster (enough to infer a theme)

SYSTEM = (
    "You name thematic clusters of vocabulary words by MEANING. Given words grouped by an "
    "algorithm, find what they share SEMANTICALLY — the concept, domain, or feeling they evoke. "
    "Look hard: many clusters have a real theme that isn't obvious at first glance (e.g. words of "
    "deception, words about gradual change, words for social status). "
    "IGNORE surface features: do NOT name a cluster after a shared suffix, prefix, spelling "
    "pattern, or part of speech (never 'words ending in -ate' or 'abstract nouns'). Those are not "
    "themes. "
    "Produce: a 'name' (2-4 words, title case, evocative, e.g. 'Deception & Concealment', "
    "'Decline & Decay', 'Praise & Devotion') and a 'desc' (under 12 words). "
    "ONLY if the words are truly semantically unrelated after genuine effort, name it "
    "'Mixed Vocabulary' and say so in the desc. Use this sparingly — it is a last resort. "
    "Respond ONLY with JSON: {\"name\":\"...\",\"desc\":\"...\"}."
)

def load(path):
    return json.load(path and open(path)) if os.path.exists(path) else {}

def main():
    vocab = json.load(open(INPUT_JSON))
    done = json.load(open(OUTPUT_JSON)) if os.path.exists(OUTPUT_JSON) else {}

    # group words by cluster
    by_cluster = defaultdict(list)
    for word, d in vocab.items():
        by_cluster[d["cluster"]].append(word)

    clusters = sorted(by_cluster.keys())
    print(f"{len(clusters)} clusters found")

    for c in clusters:
        key = str(c)
        if key in done:
            continue
        words = by_cluster[c]
        sample = words[:SAMPLE_SIZE]
        prompt = f"Cluster {c} ({len(words)} words). Sample words:\n" + ", ".join(sample)
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.4,
                )
                obj = json.loads(resp.choices[0].message.content)
                done[key] = {
                    "name": obj.get("name", f"Cluster {c}"),
                    "desc": obj.get("desc", ""),
                    "count": len(words),
                }
                json.dump(done, open(OUTPUT_JSON, "w"), ensure_ascii=False, indent=2)
                print(f"cluster {c}: {done[key]['name']} ({len(words)} words)")
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  cluster {c} failed ({e}); retry in {wait}s")
                time.sleep(wait)
        time.sleep(1)

    print(f"done — {len(done)} clusters named in {OUTPUT_JSON}")

if __name__ == "__main__":
    main()