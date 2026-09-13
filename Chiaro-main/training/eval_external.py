"""
Experiment B — external-dataset eval for the CHIARO-trained RoBERTa.

Loads the best-on-val checkpoint from expb_roberta_chiaro/best/, runs
it on each of GoEmotions / ISEAR / EmotionX-2019, scores under both
B1 (synonym-aliased) and B2 (exact-CHIARO-match) modes.

The agent-slot fill is controlled by --agent-slot:
  the_speaker  : "{utterance}</s></s>the speaker"  (variant a)
  empty        : "{utterance}</s></s>"             (variant b)
  none         : "{utterance}"                     (variant c, used after retrain)

GPU placement: CUDA_VISIBLE_DEVICES=7.
"""
import argparse, json, os, sys, urllib.request, zipfile
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "expb_roberta_chiaro", "best")
OUT_DIR  = os.path.join(HERE, "expb_external_results")
os.makedirs(OUT_DIR, exist_ok=True)

CHIARO_EMOTIONS = ['joy','pride','relief','gratitude','excitement',
                   'anger','sadness','fear','disgust','embarrassment']
CHIARO_SET = set(CHIARO_EMOTIONS)

ALIAS_MAP = {
    # exact / self-mappings (also covered by B2)
    "joy": "joy", "happy": "joy", "happiness": "joy",
    "pride": "pride", "proud": "pride",
    "relief": "relief", "relieved": "relief",
    "gratitude": "gratitude", "grateful": "gratitude", "thankful": "gratitude",
    "love": "gratitude",
    "excitement": "excitement", "excited": "excitement",
    "optimism": "excitement", "anticipation": "excitement",
    "anger": "anger", "angry": "anger",
    "annoyance": "anger", "annoyed": "anger", "frustration": "anger",
    "sadness": "sadness", "sad": "sadness",
    "grief": "sadness", "disappointment": "sadness", "remorse": "sadness",
    "fear": "fear", "afraid": "fear", "nervousness": "fear", "nervous": "fear",
    "disgust": "disgust", "disgusted": "disgust", "contempt": "disgust",
    "embarrassment": "embarrassment", "embarrassed": "embarrassment",
    "shame": "embarrassment", "ashamed": "embarrassment",
    "guilt": "embarrassment", "guilty": "embarrassment",
}


def alias_to_chiaro(label):
    """Map any external emotion label to CHIARO or return None."""
    if not label: return None
    return ALIAS_MAP.get(label.lower().strip())


# ---------- Dataset loaders ----------------------------------------

def load_goemotions():
    """GoEmotions simplified split; we use single-label items only."""
    from datasets import load_dataset
    # datasets 4.x requires namespaced repo id
    for repo in ["google-research-datasets/go_emotions", "go_emotions"]:
        try:
            ds = load_dataset(repo, "simplified")
            break
        except Exception as e:
            ds = None; last = e
    if ds is None:
        raise RuntimeError(f"Could not load GoEmotions: {last}")
    test = ds["test"]
    names = test.features["labels"].feature.names
    rows = []
    for r in test:
        labs = r["labels"]
        if len(labs) != 1: continue   # single-label items only
        rows.append({"text": r["text"], "gold": names[labs[0]].lower()})
    return rows


def load_isear():
    """ISEAR. Has no canonical split — use whatever we get. Try multiple
       HF mirrors, then fall back to a GitHub CSV mirror."""
    from datasets import load_dataset
    last = None
    for repo in ["tanfiona/ISEAR", "ma2za/isear", "Aravinth-Megnath/Isear-data",
                 "lukasgarbas/isear-emotion", "dair-ai/emotion-isear"]:
        try:
            ds = load_dataset(repo)
            split = ds.get("test") or ds.get("validation") or ds["train"]
            cols = split.column_names
            text_col = ("text" if "text" in cols else
                        ("sentence" if "sentence" in cols else
                         ("SIT" if "SIT" in cols else cols[0])))
            label_col = ("emotion" if "emotion" in cols else
                         ("label" if "label" in cols else
                          ("Field1" if "Field1" in cols else cols[1])))
            print(f"  ISEAR loaded from HF '{repo}'  columns -> text='{text_col}', label='{label_col}'")
            rows = []
            for r in split:
                lab = r[label_col]
                if isinstance(lab, int):
                    lab = split.features[label_col].names[lab]
                if not r[text_col]: continue
                rows.append({"text": str(r[text_col]), "gold": str(lab).lower().strip()})
            return rows
        except Exception as e:
            last = e

    # Fall back to GitHub CSV mirror
    print(f"  HF mirrors failed (last: {last}); falling back to GitHub CSV mirror")
    import csv, urllib.request, io
    URL = "https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/master/isear.csv"
    raw = urllib.request.urlopen(URL).read().decode("utf-8", errors="replace")
    rows = []
    rdr = csv.reader(io.StringIO(raw), delimiter="|")
    # ISEAR CSV columns: META | EMOT | SIT (the situation description)
    header = next(rdr, None)
    emot_idx = header.index("EMOT") if header and "EMOT" in header else 1
    sit_idx  = header.index("SIT")  if header and "SIT"  in header else -1
    # Map ISEAR's integer emotion codes (1=joy, 2=fear, 3=anger, 4=sadness, 5=disgust, 6=shame, 7=guilt)
    ISEAR_CODE_MAP = {1: "joy", 2: "fear", 3: "anger", 4: "sadness",
                      5: "disgust", 6: "shame", 7: "guilt"}
    for row in rdr:
        if len(row) <= max(emot_idx, sit_idx): continue
        emot = row[emot_idx].strip()
        text = row[sit_idx].strip()
        # Try int code first, fall back to string
        try:
            lab = ISEAR_CODE_MAP.get(int(emot))
        except ValueError:
            lab = emot.lower()
        if not text or not lab: continue
        rows.append({"text": text, "gold": lab})
    print(f"  ISEAR GitHub: {len(rows)} rows loaded")
    return rows


EMOTIONX_REPO_URLS = [
    "https://github.com/bshmueli/EmotionX-2019/archive/refs/heads/main.zip",
    "https://github.com/bshmueli/EmotionX-2019/archive/refs/heads/master.zip",
]

def load_meld():
    """MELD (Friends, text-only). 7-class: anger / disgust / fear / joy /
       neutral / sadness / surprise. Modern successor to EmotionLines/EmotionX."""
    from datasets import load_dataset
    last = None
    for repo in ["zhouang/meld_text_only", "ajyy/MELD_audio", "declare-lab/MELD"]:
        try:
            ds = load_dataset(repo)
            split = ds.get("test") or ds.get("validation") or ds["train"]
            cols = split.column_names
            text_col = ("Utterance" if "Utterance" in cols else
                        ("text" if "text" in cols else
                         ("utterance" if "utterance" in cols else cols[0])))
            label_col = ("Emotion" if "Emotion" in cols else
                         ("emotion" if "emotion" in cols else
                          ("label" if "label" in cols else cols[1])))
            print(f"  MELD loaded from HF '{repo}'  columns -> text='{text_col}', label='{label_col}'")
            rows = []
            for r in split:
                lab = r[label_col]
                if isinstance(lab, int):
                    lab = split.features[label_col].names[lab]
                txt = r[text_col]
                if not txt or not lab: continue
                rows.append({"text": str(txt), "gold": str(lab).lower().strip()})
            return rows
        except Exception as e:
            last = e
    # Fall back to declare-lab MELD GitHub CSV mirror
    print(f"  HF mirrors failed (last: {last}); falling back to GitHub CSV mirror")
    import csv, urllib.request, io
    URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv"
    raw = urllib.request.urlopen(URL).read().decode("utf-8", errors="replace")
    rdr = csv.DictReader(io.StringIO(raw))
    rows = []
    for r in rdr:
        txt = r.get("Utterance") or r.get("utterance")
        lab = r.get("Emotion") or r.get("emotion")
        if not txt or not lab: continue
        rows.append({"text": txt, "gold": lab.lower().strip()})
    print(f"  MELD GitHub: {len(rows)} rows loaded")
    return rows

def load_dailydialog():
    """DailyDialog (Li et al., IJCNLP 2017): per-turn emotion in everyday-life
       conversations. 7-class: no_emotion, anger, disgust, fear, happiness,
       sadness, surprise. 4 exact CHIARO matches (anger, disgust, fear,
       sadness) for B2 scoring.

       Loads via HF's auto-converted parquet ref (the script-based mirror is
       deprecated in the new `datasets` library; the original ZIP requires
       a JS fingerprint check)."""
    import pandas as pd, urllib.request
    PARQUET_URL = ("https://huggingface.co/datasets/li2017dailydialog/daily_dialog/"
                   "resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet")
    local = os.path.join(HERE, "_dailydialog_test.parquet")
    if not os.path.exists(local):
        urllib.request.urlretrieve(PARQUET_URL, local)
    df = pd.read_parquet(local)
    CANONICAL = ["no_emotion","anger","disgust","fear","happiness",
                 "sadness","surprise"]
    rows = []
    for _, r in df.iterrows():
        for u, e in zip(r["dialog"], r["emotion"]):
            e = int(e)
            if 0 <= e < len(CANONICAL):
                name = CANONICAL[e]
                if name != "no_emotion":
                    rows.append({"text": str(u).strip(), "gold": name})
    print(f"  DailyDialog (parquet) loaded: rows={len(rows)}")
    return rows


def _OLD_load_dailydialog():
    """Old stub kept for reference."""
    from datasets import load_dataset

    # Try a parquet-based HF mirror first (no script needed).
    for repo in ["mteb/DailyDialog", "benjamin/dailydialog", "ahsanqayum/DailyDialog"]:
        try:
            ds = load_dataset(repo)
            split = ds.get("test") or ds.get("validation") or ds["train"]
            cols = split.column_names
            text_col = ("utterance" if "utterance" in cols else
                        ("text" if "text" in cols else
                         ("Utterance" if "Utterance" in cols else cols[0])))
            label_col = ("emotion" if "emotion" in cols else
                         ("label" if "label" in cols else
                          ("Emotion" if "Emotion" in cols else cols[1])))
            rows = []
            for r in split:
                lab = r[label_col]
                if isinstance(lab, int):
                    try:
                        lab = split.features[label_col].names[lab]
                    except Exception:
                        pass
                if not r[text_col]: continue
                lab = str(lab).lower().strip()
                if lab in ("no_emotion","neutral",""):
                    continue
                rows.append({"text": str(r[text_col]), "gold": lab})
            print(f"  DailyDialog loaded from HF '{repo}' (parquet)  rows={len(rows)}")
            return rows
        except Exception:
            pass

    # Fall back to original ZIP from the DailyDialog author's hosted file.
    import urllib.request, zipfile, io
    URLS = [
        "http://yanran.li/files/ijcnlp_dailydialog.zip",
        # GitHub mirror (some forks)
        "https://huggingface.co/datasets/li2017dailydialog/daily_dialog/resolve/main/ijcnlp_dailydialog.zip",
    ]
    local_zip = os.path.join(HERE, "_dailydialog.zip")
    if not os.path.exists(local_zip):
        err = None
        for url in URLS:
            try:
                urllib.request.urlretrieve(url, local_zip); err = None; break
            except Exception as e:
                err = e
        if err:
            raise RuntimeError(f"could not download DailyDialog: {err}")

    # Extract dialogues_test.txt and dialogues_emotion_test.txt
    with zipfile.ZipFile(local_zip) as zf:
        names = zf.namelist()
        test_utts = next((n for n in names if "dialogues_test.txt" in n and "emotion" not in n and "act" not in n and "topic" not in n), None)
        test_emos = next((n for n in names if "dialogues_emotion_test.txt" in n), None)
        if not test_utts or not test_emos:
            raise RuntimeError(f"could not find test files in zip; got names: {names[:30]}")
        utts_raw = zf.read(test_utts).decode("utf-8", errors="replace")
        emos_raw = zf.read(test_emos).decode("utf-8", errors="replace")

    CANONICAL = ["no_emotion","anger","disgust","fear","happiness","sadness","surprise"]
    rows = []
    for u_line, e_line in zip(utts_raw.splitlines(), emos_raw.splitlines()):
        utts = [x.strip() for x in u_line.split("__eou__") if x.strip()]
        emos = [int(x.strip()) for x in e_line.split() if x.strip().isdigit()]
        for u, e in zip(utts, emos):
            if 0 <= e < len(CANONICAL):
                name = CANONICAL[e]
                if name != "no_emotion":
                    rows.append({"text": u, "gold": name})
    print(f"  DailyDialog loaded from local ZIP {local_zip}  rows={len(rows)}")
    return rows


def load_xed():
    """XED (Öhman et al., COLING 2020): cross-lingual emotion dataset from
       movie subtitles. We use the English-annotated split. 8-class Plutchik
       (anger, anticipation, disgust, fear, joy, sadness, surprise, trust);
       multi-label per line. 5 exact CHIARO matches (anger, disgust, fear,
       joy, sadness) — we keep lines whose label set contains a CHIARO label
       and use that as the gold; lines with two CHIARO labels are dropped.

       Loaded via HF auto-convert parquet ref (the script mirror is
       deprecated)."""
    import urllib.request, pandas as pd
    PARQUET_URL = ("https://huggingface.co/datasets/Helsinki-NLP/xed_en_fi/"
                   "resolve/refs%2Fconvert%2Fparquet/en_annotated/train/0000.parquet")
    local = os.path.join(HERE, "_xed_en.parquet")
    if not os.path.exists(local):
        urllib.request.urlretrieve(PARQUET_URL, local)
    df = pd.read_parquet(local)
    # XED label IDs (Plutchik 8, indices from the README):
    PLUTCHIK = {1: "anger", 2: "anticipation", 3: "disgust", 4: "fear",
                5: "joy", 6: "sadness", 7: "surprise", 8: "trust"}
    CHIARO = {"joy","pride","relief","gratitude","excitement",
              "anger","sadness","fear","disgust","embarrassment"}
    rows = []
    for _, r in df.iterrows():
        labs = list(r["labels"])
        chiaro_hits = [PLUTCHIK[l] for l in labs if l in PLUTCHIK and PLUTCHIK[l] in CHIARO]
        if len(chiaro_hits) != 1:
            continue
        text = str(r["sentence"]).strip()
        if not text: continue
        rows.append({"text": text, "gold": chiaro_hits[0]})
    print(f"  XED (English, parquet) loaded: rows={len(rows)} (single-CHIARO-label only)")
    return rows


def load_semeval2018():
    """SemEval-2018 Task 1 (Affect in Tweets), Subtask 5 English (Mohammad
       et al., 2018, *SEM workshop). 11-class multi-label. We keep items
       with exactly one label, and only when that label is a CHIARO exact
       match (anger, disgust, fear, joy, sadness)."""
    import urllib.request, pandas as pd
    PARQUET_URL = ("https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2018_task_1/"
                   "resolve/refs%2Fconvert%2Fparquet/subtask5.english/test/0000.parquet")
    local = os.path.join(HERE, "_semeval2018_test.parquet")
    if not os.path.exists(local):
        urllib.request.urlretrieve(PARQUET_URL, local)
    df = pd.read_parquet(local)
    CHIARO_COLS = ["anger","disgust","fear","joy","sadness"]
    ALL_EMO_COLS = ["anger","anticipation","disgust","fear","joy","love",
                    "optimism","pessimism","sadness","surprise","trust"]
    rows = []
    for _, r in df.iterrows():
        hits = [c for c in ALL_EMO_COLS if bool(r[c])]
        if len(hits) != 1:
            continue
        if hits[0] not in CHIARO_COLS:
            continue
        rows.append({"text": str(r["Tweet"]).strip(), "gold": hits[0]})
    print(f"  SemEval-2018 (English test, parquet) loaded: rows={len(rows)} (single-CHIARO-label only)")
    return rows


def load_emobench_eu(use_subject_as_role=True):
    """EmoBench EU (Sabour et al., ACL 2024): theory-of-mind emotion
       attribution. Each item gives a scenario (narrative text) and a
       subject (the character whose emotion we predict). 200 English items.

       Structurally the closest existing dataset to CHIARO: third-person
       agent-attributed emotion with a noun subject. Unlike all other
       transfer-out benchmarks where the role slot is filled with the
       generic "the speaker", here the role slot is filled with the actual
       subject — so this is the natural test of whether CHIARO-style agent
       conditioning transfers to a paper-authored attribution benchmark.

       Returns rows with a special 'role' field overriding the agent-slot
       CLI flag."""
    import urllib.request
    URL = "https://raw.githubusercontent.com/Sahandfer/EmoBench/master/data/EU.jsonl"
    local = os.path.join(HERE, "_emobench_eu.jsonl")
    if not os.path.exists(local):
        urllib.request.urlretrieve(URL, local)
    rows = []
    for line in open(local, encoding="utf-8"):
        r = json.loads(line)
        if r.get("language") != "en": continue
        rows.append({
            "text": r["scenario"],
            "gold": r["emotion_label"].lower().strip(),
            "role": r["subject"] if use_subject_as_role else None,
        })
    print(f"  EmoBench EU (English) loaded: rows={len(rows)}")
    return rows


def load_tweeteval():
    """TweetEval emotion subset (Barbieri et al., Findings of EMNLP 2020).
       Wraps SemEval-2018 Affect-in-Tweets, 4-class:
       anger, joy, optimism, sadness. 3 exact CHIARO matches (anger, joy,
       sadness) for B2 scoring."""
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "emotion")
    split = ds.get("test") or ds["train"]
    names = split.features["label"].names
    rows = []
    for r in split:
        rows.append({"text": str(r["text"]), "gold": names[r["label"]].lower()})
    print(f"  TweetEval emotion (parquet) loaded: rows={len(rows)}")
    return rows


def load_carer():
    """CARER tweet emotion dataset (HF: dair-ai/emotion).
       6-class: sadness, joy, love, anger, fear, surprise.
       4 exact CHIARO matches (sadness, joy, anger, fear) for B2 scoring."""
    from datasets import load_dataset
    last = None
    for repo in ["dair-ai/emotion"]:
        for config in ["split", "unsplit", None]:
            try:
                ds = load_dataset(repo) if config is None else load_dataset(repo, config)
                split = ds.get("test") or ds.get("validation") or ds["train"]
                cols = split.column_names
                text_col = "text" if "text" in cols else cols[0]
                label_col = "label" if "label" in cols else cols[1]
                rows = []
                for r in split:
                    lab = r[label_col]
                    if isinstance(lab, int):
                        lab = split.features[label_col].names[lab]
                    if not r[text_col]: continue
                    rows.append({"text": str(r[text_col]), "gold": str(lab).lower().strip()})
                print(f"  CARER loaded from HF '{repo}' (config={config})  rows={len(rows)}")
                return rows
            except Exception as e:
                last = e
    raise RuntimeError(f"CARER not loadable; last error: {last}")


def load_emotionx():
    """EmotionX-2019 Friends test JSON -- text-only utterance × emotion.

    Repo structure: nested zips. We unpack the outer repo zip, then the
    inner `2019_Eval_Labeled.zip` which contains the labeled test
    dialogues for Friends + EmotionPush.
    """
    # 1. Get the outer repo unpacked (main or master branch)
    candidates = ["EmotionX-2019-main", "EmotionX-2019-master"]
    local = None
    for c in candidates:
        full = os.path.join(HERE, c)
        if os.path.isdir(full):
            local = full; break
    if local is None:
        zpath = os.path.join(HERE, "_emotionx.zip")
        if not os.path.exists(zpath):
            err = None
            for url in EMOTIONX_REPO_URLS:
                try:
                    urllib.request.urlretrieve(url, zpath); err = None; break
                except Exception as e:
                    err = e
            if err: raise err
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(HERE)
        for c in candidates:
            full = os.path.join(HERE, c)
            if os.path.isdir(full):
                local = full; break
        if local is None:
            raise RuntimeError(f"could not find EmotionX-2019 unpacked dir; tried {candidates}")

    # 2. Unpack the inner labeled-eval zip (contains Friends + EmotionPush test sets)
    eval_zip = os.path.join(local, "2019_Eval_Labeled.zip")
    eval_dir = os.path.join(local, "_eval_labeled")
    if not os.path.isdir(eval_dir):
        if not os.path.exists(eval_zip):
            raise RuntimeError(f"missing inner zip {eval_zip}")
        with zipfile.ZipFile(eval_zip) as zf:
            zf.extractall(eval_dir)

    # 3. Locate any Friends test JSON inside the unpacked tree
    found = None
    for root, _, files in os.walk(eval_dir):
        for f in files:
            if f.endswith(".json") and "friends" in f.lower():
                found = os.path.join(root, f); break
        if found: break
    if not found:
        raise RuntimeError(f"could not find Friends JSON under {eval_dir}; "
                           f"available: {os.listdir(eval_dir)}")
    print(f"  EmotionX loaded from {found}")
    data = json.load(open(found, encoding="utf-8"))

    rows = []
    for convo in data:
        for turn in convo:
            utt = turn.get("utterance") or turn.get("text") or turn.get("sentence")
            emo = turn.get("emotion") or turn.get("label")
            if not utt or not emo: continue
            rows.append({"text": str(utt), "gold": str(emo).lower().strip()})
    return rows


# ---------- Eval --------------------------------------------------

def build_input(utterance, agent_slot_mode):
    if agent_slot_mode == "the_speaker":
        return utterance, "the speaker"
    elif agent_slot_mode == "empty":
        return utterance, ""
    elif agent_slot_mode == "none":
        return utterance, None
    else:
        raise ValueError(agent_slot_mode)


@torch.no_grad()
def predict_batch(texts_pair, tok, mdl, agent_slot_mode, device, batch=64):
    """Run model on (text1, text2 OR None) pairs, return list of CHIARO labels."""
    preds = []
    for i in range(0, len(texts_pair), batch):
        batch_rows = texts_pair[i:i+batch]
        if agent_slot_mode == "none":
            texts1 = [t1 for (t1, _) in batch_rows]
            enc = tok(texts1, truncation=True, max_length=256, padding=True,
                      return_tensors="pt").to(device)
        else:
            texts1 = [t1 for (t1, _) in batch_rows]
            texts2 = [t2 for (_, t2) in batch_rows]
            enc = tok(texts1, texts2, truncation=True, max_length=256, padding=True,
                      return_tensors="pt").to(device)
        logits = mdl(**enc).logits
        ids = logits.argmax(dim=-1).cpu().tolist()
        preds.extend([CHIARO_EMOTIONS[i] for i in ids])
    return preds


def score(rows, preds, mode):
    """Score under B1 (synonym) or B2 (exact). Return dict of metrics."""
    assert mode in ("B1", "B2")
    pool = []
    for r, p in zip(rows, preds):
        gold = r["gold"]
        if mode == "B2":
            if gold not in CHIARO_SET:
                continue
            mapped = gold
        else:  # B1
            mapped = alias_to_chiaro(gold)
            if mapped is None:
                continue
        pool.append((mapped, p, gold))
    if not pool:
        return {"n": 0, "acc": 0.0, "pred_dist": {}, "gold_dist": {}}
    correct = sum(1 for m, p, _ in pool if m == p)
    return {
        "n": len(pool),
        "acc": correct / len(pool),
        "pred_dist": dict(Counter(p for _, p, _ in pool).most_common()),
        "gold_dist_mapped": dict(Counter(m for m, _, _ in pool).most_common()),
        "gold_dist_native": dict(Counter(g for _, _, g in pool).most_common(15)),
    }


# ---------- Main --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-slot", required=True,
                    choices=["the_speaker", "empty", "none"])
    ap.add_argument("--datasets", default="goemotions,isear,emotionx",
                    help="comma-separated subset of {goemotions, isear, emotionx}")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--out-suffix", default="", help="Optional suffix appended before .json in output filename.")
    args = ap.parse_args()

    print(f"agent_slot mode: {args.agent_slot}")
    print(f"checkpoint:     {args.ckpt_dir}")
    print(f"datasets:       {args.datasets}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.ckpt_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.ckpt_dir).to(device).eval()
    print(f"loaded model on {device}")

    LOADERS = {"goemotions": load_goemotions, "isear": load_isear,
               "emotionx": load_emotionx, "meld": load_meld,
               "dailydialog": load_dailydialog, "carer": load_carer,
               "tweeteval": load_tweeteval, "emobench_eu": load_emobench_eu,
               "xed": load_xed, "semeval2018": load_semeval2018}
    todo = [s.strip() for s in args.datasets.split(",") if s.strip()]

    all_results = {}
    for name in todo:
        print(f"\n=== {name} ===")
        rows = LOADERS[name]()
        print(f"  loaded {len(rows)} items (raw native labels)")
        # Native label distribution (top 10)
        native_dist = Counter(r["gold"] for r in rows).most_common(10)
        print(f"  native top-10: {native_dist}")

        # Per-row 'role' (e.g. EmoBench's subject) overrides --agent-slot.
        def _pair(r):
            if r.get("role") is not None:
                return (r["text"], r["role"])
            return build_input(r["text"], args.agent_slot)
        texts_pair = [_pair(r) for r in rows]
        # Force pair-input path when any row supplies an explicit role.
        slot_mode = args.agent_slot
        if any(r.get("role") is not None for r in rows):
            slot_mode = "the_speaker"
        preds = predict_batch(texts_pair, tok, mdl, slot_mode, device)
        print(f"  preds done. distribution: {dict(Counter(preds).most_common())}")

        # Score under both modes
        b1 = score(rows, preds, "B1")
        b2 = score(rows, preds, "B2")
        # majority-class baseline on each pool
        b1_mc = max(b1.get("gold_dist_mapped", {}).values(), default=0) / max(b1["n"], 1)
        b2_mc = max(b2.get("gold_dist_mapped", {}).values(), default=0) / max(b2["n"], 1)

        print(f"  B1 (synonym):   n={b1['n']:5d}  acc={b1['acc']*100:5.1f}%  majority-class={b1_mc*100:5.1f}%")
        print(f"  B2 (exact):     n={b2['n']:5d}  acc={b2['acc']*100:5.1f}%  majority-class={b2_mc*100:5.1f}%")

        # Top-pick concentration check (degenerate threshold = 60%)
        top_pick_share_b1 = max((b1["pred_dist"].values()), default=0) / max(b1["n"], 1)
        top_pick_share_b2 = max((b2["pred_dist"].values()), default=0) / max(b2["n"], 1)
        print(f"  top-pick share (degenerate if >60%):  B1={top_pick_share_b1*100:.1f}%   B2={top_pick_share_b2*100:.1f}%")

        all_results[name] = {
            "agent_slot": args.agent_slot,
            "B1_synonym": b1, "B1_majority_class_baseline": b1_mc,
            "B2_exact":  b2,  "B2_majority_class_baseline": b2_mc,
            "n_raw_items": len(rows),
        }

    out_path = os.path.join(OUT_DIR, f"results_agent={args.agent_slot}{args.out_suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
