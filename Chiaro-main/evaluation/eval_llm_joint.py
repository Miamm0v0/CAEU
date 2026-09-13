"""Reference LLM evaluation runner for CHIARO (joint two-agent MCQ protocol).

Runs the exact prompt from the paper (Appendix: Evaluation Prompt) against any
OpenAI-compatible chat-completions endpoint and writes prediction records that
score_predictions.py can consume.

Examples:
  # OpenAI
  OPENAI_API_KEY=... python eval_llm_joint.py --model gpt-5.5 --out raw_predictions/llm/eval_my_model.json
  # Any OpenAI-compatible endpoint (Together, DeepSeek, vLLM, ...)
  API_KEY=... python eval_llm_joint.py --model Qwen/Qwen3.5-27B \
      --base-url https://api.together.xyz/v1 --key-env API_KEY --out my_preds.json

The run is resume-safe: re-running with the same --out only queries missing scenes.
"""
import argparse
import io
import json
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "chiaro_full.json")

SYSTEM = (
    "You are answering a multiple-choice question about a sentence that describes "
    "two people reacting to the same event with contrasting emotions.\n\n"
    "Read the sentence carefully and select the single best answer for each agent.\n"
    "Reply with EXACTLY two lines in this format:\nAGENT A: <letter>\nAGENT B: <letter>\nNothing else."
)


def build_user(item):
    opts_a = "\n".join(f"{k}. {v}" for k, v in sorted(item["options_a"].items()))
    opts_b = "\n".join(f"{k}. {v}" for k, v in sorted(item["options_b"].items()))
    la = ", ".join(sorted(item["options_a"].keys()))
    lb = ", ".join(sorted(item["options_b"].keys()))
    return (f"SENTENCE: {item['sentence']}\n\nAGENT A: {item['agent_a_role']}\n"
            f"AGENT B: {item['agent_b_role']}\n\nQUESTION: What emotion does each agent feel?\n\n"
            f"AGENT A OPTIONS:\n{opts_a}\n\nAGENT B OPTIONS:\n{opts_b}\n\n"
            f"Reply with exactly two lines:\nAGENT A: <letter from {la}>\nAGENT B: <letter from {lb}>")


def parse_joint(raw):
    if not raw:
        return None, None
    a = re.search(r"AGENT\s*A[^A-Za-z]*([A-E])\b", raw, re.IGNORECASE)
    b = re.search(r"AGENT\s*B[^A-Za-z]*([A-E])\b", raw, re.IGNORECASE)
    return (a.group(1).upper() if a else None), (b.group(1).upper() if b else None)


def call(base_url, key, model, item, max_retries=4):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": build_user(item)}],
        "temperature": 0,
        "max_tokens": 40,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(f"{base_url}/chat/completions", timeout=120,
                              headers={"Authorization": f"Bearer {key}"}, json=body)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--key-env", default="OPENAI_API_KEY", help="env var holding the API key")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N scenes")
    args = ap.parse_args()

    key = os.environ.get(args.key_env)
    if not key:
        sys.exit(f"set {args.key_env} in the environment")

    items = json.load(io.open(DATA, encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    done = {}
    if os.path.exists(args.out):
        done = {r["mcq_id_a"]: r for r in json.load(io.open(args.out, encoding="utf-8"))}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} scenes | {len(done)} cached | {len(todo)} to run", flush=True)

    for n, it in enumerate(todo, 1):
        raw = call(args.base_url, key, args.model, it)
        la, lb = parse_joint(raw)
        done[it["id"]] = {
            "mcq_id_a": it["id"],
            "sentence": it["sentence"],
            "agent_a_role": it["agent_a_role"], "agent_b_role": it["agent_b_role"],
            "options_a": it["options_a"], "options_b": it["options_b"],
            "gold_A": it["generation_emotion_a"], "gold_B": it["generation_emotion_b"],
            "llm_letter_A": la, "llm_emotion_A": it["options_a"].get(la) if la else None,
            "llm_letter_B": lb, "llm_emotion_B": it["options_b"].get(lb) if lb else None,
            "raw": raw, "model": args.model, "version": it.get("version"),
        }
        if n % 25 == 0 or n == len(todo):
            rows = sorted(done.values(), key=lambda r: str(r["mcq_id_a"]))
            tmp = args.out + ".tmp"
            json.dump(rows, io.open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            os.replace(tmp, args.out)
            print(f"  {n}/{len(todo)}", flush=True)

    print(f"wrote {args.out}; score it with:  python score_predictions.py")


if __name__ == "__main__":
    main()
