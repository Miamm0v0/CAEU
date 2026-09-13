"""
noun_count_validator.py

Experiment 1: flag sentences whose number of nouns / distinct people
exceeds a threshold so the generation pipeline can regenerate them.

The annotator's pain point isn't generic "many nouns" (every sentence
mentions tables, doors, etc.); it's "many people / relations" — too many
distinct characters to keep straight. This validator therefore reports
two metrics:

  - all_nouns: every word NLTK tags NN / NNS / NNP / NNPS
  - person_refs: distinct entities referring to a PERSON (proper nouns
    + common person-role nouns like "mother", "boyfriend",
    "neighbor", "boss"); duplicates merged

The default flag rule is `person_refs > 3` (matches the user's "> 3"
threshold for the meaningful axis). All-noun count is shown alongside
for transparency.

Run:
  python noun_count_validator.py                  # default: 4-13/stories_v2
  STORIES_DIR=../4-23/stories_v2_balanced python noun_count_validator.py
  THRESHOLD=4 python noun_count_validator.py
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import os
import re
from collections import Counter
from glob import glob

import nltk
# Make sure tagger data is present (no-op if already downloaded)
for pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
            "punkt", "punkt_tab"):
    try:
        nltk.data.find(pkg)
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except Exception: pass

from nltk import pos_tag, word_tokenize

# ── Config ────────────────────────────────────────────────────────────────────

STORIES_DIR = os.getenv("STORIES_DIR", "../4-13/stories_v2")
THRESHOLD = int(os.getenv("THRESHOLD", "3"))

# Common nouns that refer to a person (kinship, role, etc.). Tokens are
# matched case-insensitively. Anything matching this list adds 1 to the
# person count even if it's a common noun.
PERSON_ROLE_NOUNS = {
    # family
    "mother", "mom", "mum", "father", "dad", "son", "daughter",
    "sister", "brother", "wife", "husband", "spouse", "partner",
    "fiance", "fiancé", "fiancee", "fiancée", "boyfriend", "girlfriend",
    "aunt", "uncle", "cousin", "niece", "nephew",
    "grandmother", "grandma", "grandfather", "grandpa",
    "grandchild", "grandson", "granddaughter",
    "parent", "parents", "child", "kid", "baby", "toddler", "teen",
    "teenager", "adult",
    "stepmother", "stepfather", "stepson", "stepdaughter", "stepsister",
    "stepbrother", "stepkid",
    # social
    "friend", "neighbor", "neighbour", "roommate", "classmate",
    "colleague", "coworker", "boss", "manager", "employee", "employer",
    "customer", "client", "guest", "host", "stranger", "visitor",
    # professional / generic person
    "student", "teacher", "instructor", "lawyer", "doctor", "nurse",
    "trainee", "trainer", "engineer", "rep", "tenant", "shopper",
    "groomer", "guitarist", "musician", "baker", "bride", "groom",
    "father-in-law", "mother-in-law",
    # the other person
    "person", "man", "woman", "girl", "boy", "lady", "gentleman",
    "guy", "people",
}


def is_proper_noun(tag: str) -> bool:
    return tag in ("NNP", "NNPS")


def is_noun(tag: str) -> bool:
    return tag in ("NN", "NNS", "NNP", "NNPS")


_TOKEN_BLACKLIST_RE = re.compile(r"^[\W_]+$|^[a-zA-Z]\.?$")  # punctuation-only or single letter
_HONORIFICS = {"mr", "mrs", "ms", "dr", "prof"}

# Capitalised tokens that look like proper nouns to NLTK but are NOT people.
# Filtered out before counting person references.
NON_PERSON_PROPER_NOUNS = {
    # Countries / regions / nationalities / languages
    "america","american","england","english","britain","british","uk","u.k.","us","u.s.",
    "belgium","belgian","dutch","german","germany","french","france",
    "italian","italy","spanish","spain","chinese","china","japan","japanese",
    "indian","india","mexican","mexico","canada","canadian","texas","new","york",
    "european","europe","asia","asian","african","africa","australian","australia",
    "scottish","scotland","irish","ireland","welsh","wales","russian","russia",
    "korea","korean","greek","greece","polish","poland",
    # Cities / common locations
    "london","paris","tokyo","beijing","sydney","toronto","berlin","madrid","rome",
    # Holidays / months / events
    "christmas","thanksgiving","easter","halloween","diwali","ramadan","hanukkah",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
    "january","february","march","april","may","june","july","august",
    "september","october","november","december",
    "summer","winter","spring","fall","autumn",
    # Acronyms commonly seen in AITA stories
    "ocd","aita","esl","iep","uhc","hr","irs","cpr","mil","sil","fil","wfh",
    "etsy","amazon","facebook","instagram","tiktok","reddit","twitter","x",
    "rsvp","gps","atm","dvd","cd","tv","cctv","mri","resp","ipad","iphone","mac",
    # Sign / quoted-text / common phrase artefacts
    "private","do","enter","exit","open","closed","stop","go","yes","no",
    "warning","caution","danger","welcome","sale","sold","free","new",
    # Brand / product names treated as persons
    "budweiser","coca","cola","pepsi","starbucks","mcdonald","mcdonald's",
    # Religion / generic identifiers
    "god","allah","jesus","muslim","christian","jewish","buddhist","hindu",
}

def _good_token(w: str) -> bool:
    if _TOKEN_BLACKLIST_RE.match(w): return False
    if w.lower().rstrip(".") in _HONORIFICS: return False
    if w.lower() in NON_PERSON_PROPER_NOUNS: return False
    return True

def analyze(sentence: str):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)

    all_nouns = [w for w, t in tagged if is_noun(t) and _good_token(w)]
    proper_nouns = {w for w, t in tagged if is_proper_noun(t) and _good_token(w)}

    # Person references: proper nouns + common person-role tokens (deduped)
    person_set = set(proper_nouns)
    for w, t in tagged:
        if not _good_token(w): continue
        if is_noun(t) and w.lower() in PERSON_ROLE_NOUNS:
            person_set.add(w.lower())

    return {
        "all_noun_count": len(all_nouns),
        "all_nouns": all_nouns,
        "proper_nouns": sorted(proper_nouns),
        "person_refs": sorted(person_set),
        "person_count": len(person_set),
    }


def iter_sentences_in_story(story: dict):
    # Two-version stories
    for ver in ("physical_version", "non_physical_version"):
        blob = story.get(ver)
        if isinstance(blob, dict) and blob.get("sentence"):
            yield ver, blob["sentence"]
    # Single-version story (flat schema)
    if "sentence" in story and "physical_version" not in story:
        yield "single", story["sentence"]


def main() -> None:
    if not os.path.isdir(STORIES_DIR):
        raise SystemExit(f"Directory not found: {STORIES_DIR}")

    files = sorted(glob(os.path.join(STORIES_DIR, "story_*.json")))
    if not files:
        raise SystemExit(f"No story_*.json files in {STORIES_DIR}")

    flagged = []
    all_results = []
    person_counts = Counter()
    noun_counts = Counter()

    for fp in files:
        try:
            story = json.load(open(fp, encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {fp}: {e}")
            continue
        for ver, sent in iter_sentences_in_story(story):
            r = analyze(sent)
            r["file"] = os.path.basename(fp)
            r["version"] = ver
            r["sentence"] = sent
            all_results.append(r)
            person_counts[r["person_count"]] += 1
            noun_counts[r["all_noun_count"]] += 1
            if r["person_count"] > THRESHOLD:
                flagged.append(r)

    print(f"Source: {STORIES_DIR}")
    print(f"Sentences analyzed: {len(all_results)}")
    print(f"Person-reference threshold (flag if >): {THRESHOLD}\n")

    # Distribution
    print("Distribution of person references per sentence:")
    for k in sorted(person_counts):
        bar = "#" * person_counts[k]
        marker = " <-- FLAG" if k > THRESHOLD else ""
        print(f"  {k:>2} people: {person_counts[k]:>3} sentences  {bar}{marker}")
    print()

    print("Distribution of all-noun counts per sentence (for reference):")
    keys = sorted(noun_counts)
    for k in keys:
        if k % 2 == 0 or k == max(keys):
            print(f"  {k:>2} nouns: {noun_counts[k]:>3} sentences")
    print()

    print(f"FLAGGED (>{THRESHOLD} people): {len(flagged)} / {len(all_results)} ({100*len(flagged)/len(all_results):.1f}%)\n")

    print("Examples of flagged sentences:")
    for r in flagged[:8]:
        print(f"  [{r['version']}] {r['file'][:55]}")
        print(f"    person_refs ({r['person_count']}): {r['person_refs']}")
        print(f"    sentence: {r['sentence'][:200]}")
        print()

    if not flagged:
        print("No sentences exceed the threshold.")

    out_path = "noun_validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "stories_dir": STORIES_DIR,
            "threshold_person_refs": THRESHOLD,
            "n_sentences": len(all_results),
            "n_flagged": len(flagged),
            "person_count_distribution": dict(sorted(person_counts.items())),
            "noun_count_distribution": dict(sorted(noun_counts.items())),
            "flagged": flagged,
            "all": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Detailed report saved -> {out_path}")


if __name__ == "__main__":
    main()
