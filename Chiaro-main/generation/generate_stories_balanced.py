"""
generate_stories_balanced.py

Distribution-fixing variant of generate_stories_v2. Instead of letting the
LLM pick whichever emotions fit the AITA post, we:

  1. Pre-assign a target (pos, neg) emotion pair per story — the assignment
     list is tuned to fill gaps in the current v2_new distribution
     (gratitude 0, disgust 0, etc.).
  2. Force those targets in the render prompt as MANDATORY constraints.
  3. Optionally filter the Reddit dump for keywords matching the target
     emotion (e.g., "thanked" / "helped" for gratitude) so the AITA
     inspiration is amenable.
  4. Reject the generated version if the emotion doesn't match the target
     and retry with a stronger prompt; only accept when the target is met.

Writes to 4-23/stories_v2_balanced/ by default.

Run:
  python generate_stories_balanced.py
Env overrides:
  NUM_STORIES       (default 15)
  OUTPUT_DIR        (default stories_v2_balanced)
  STORY_IDX_BASE    (default auto-detects max existing index in OUTPUT_DIR)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import os
import random
import re
from collections import Counter
from glob import glob as _glob

# Reuse everything from the hardened two-version generator
from generate_stories_v2 import (
    client,
    CONTRASTIVE_CATEGORIES,
    DRAFT_PROMPT_TEMPLATE,
    DRAFT_SCHEMA,
    _RENDER_CORE,
    _REPAIR_SYSTEM,
    PHYS_SCHEMA,
    NONPHYS_SCHEMA,
    POSITIVE,
    NEGATIVE,
    schema_call,
    validate_version,
    _build_category_block,
)

# ── Config ────────────────────────────────────────────────────────────────────

NUM_STORIES = int(os.getenv("NUM_STORIES", "15"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "stories_v2_balanced")
WORKERS = int(os.getenv("WORKERS", "1"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

REDDIT_DUMP = os.getenv("REDDIT_DUMP", os.path.join("..", "reddit", "AmItheAsshole_reddit_dump.jsonl"))

# ── Target emotion pairs — 20%-even distribution ──────────────────────────────
#
# For the 5-14 round we want every positive emotion to appear in ~20% of the
# stories and every negative emotion to appear in ~20%. We cycle through all
# 25 (positive × negative) pairs in round-robin order and slice to NUM_STORIES.
# This is mathematically the most-even allocation: each emotion ends up in
# exactly NUM_STORIES/5 stories whenever NUM_STORIES is a multiple of 5.

POSITIVE_EMOS = ["joy", "pride", "relief", "gratitude", "excitement"]
NEGATIVE_EMOS = ["anger", "sadness", "fear", "disgust", "embarrassment"]
_ALL_PAIRS = [(p, n) for p in POSITIVE_EMOS for n in NEGATIVE_EMOS]  # 25 pairs

_TARGET_PAIRS_FILE = os.getenv("TARGET_PAIRS_FILE", "")
if _TARGET_PAIRS_FILE and os.path.exists(_TARGET_PAIRS_FILE):
    with open(_TARGET_PAIRS_FILE, encoding="utf-8") as f:
        TARGET_PAIRS = [tuple(p) for p in json.load(f)]
    print(f"[target-pairs] loaded {len(TARGET_PAIRS)} from {_TARGET_PAIRS_FILE}")
    if len(TARGET_PAIRS) != NUM_STORIES:
        print(f"[target-pairs] overriding NUM_STORIES {NUM_STORIES} -> {len(TARGET_PAIRS)}")
        NUM_STORIES = len(TARGET_PAIRS)
else:
    TARGET_PAIRS: list[tuple[str, str]] = []
    for i in range(NUM_STORIES):
        TARGET_PAIRS.append(_ALL_PAIRS[i % len(_ALL_PAIRS)])

# ── Reddit post filtering by keyword per target emotion ──────────────────────
# For emotions the LLM tends to avoid, filter the AITA dump for posts whose
# body naturally contains related keywords. Increases hit rate, reduces
# retries.

EMOTION_REDDIT_KEYWORDS: dict[str, list[str]] = {
    "gratitude":  ["thanked", "thank you", "helped me", "saved me", "paid for",
                   "picked me up", "covered the", "gave me a ride", "lent me"],
    "disgust":    ["disgusted", "revolting", "gross", "bigot", "cheated",
                   "racist", "lied to", "stole from", "degrading"],
    "excitement": ["can't wait", "looking forward", "about to", "next week",
                   "upcoming", "planning a trip"],
    "relief":     ["worried", "scared that", "afraid", "stress", "dreading"],
    "fear":       ["might", "could happen", "worried", "scared"],
    "embarrassment": ["in front of", "publicly", "in the middle of everyone",
                      "at the party", "at the wedding"],
    "joy":        ["won", "got the", "received", "bought the"],
    "pride":      ["worked hard", "for months", "practiced", "studied",
                   "built", "finished", "completed"],
    "anger":      ["unfair", "ridiculous", "wrong of them", "disrespected"],
    "sadness":    ["missed out", "couldn't", "lost", "had to miss"],
}


_EXCLUDED_POST_IDS: set[str] | None = None


def _load_excluded_post_ids() -> set[str]:
    """Load post IDs to skip from EXCLUDED_POSTS_FILE env var (lazy, cached)."""
    global _EXCLUDED_POST_IDS
    if _EXCLUDED_POST_IDS is not None:
        return _EXCLUDED_POST_IDS
    path = os.getenv("EXCLUDED_POSTS_FILE", "")
    if not path or not os.path.exists(path):
        _EXCLUDED_POST_IDS = set()
        return _EXCLUDED_POST_IDS
    with open(path, encoding="utf-8") as f:
        _EXCLUDED_POST_IDS = set(json.load(f))
    print(f"[exclude] loaded {len(_EXCLUDED_POST_IDS)} post ids from {path}")
    return _EXCLUDED_POST_IDS


def filter_reddit_posts(lines: list[str], target_pos: str, target_neg: str,
                        n_wanted: int = 80) -> list[dict]:
    """Return Reddit posts that contain at least one keyword from either target.

    Posts whose `id` is in EXCLUDED_POSTS_FILE (if set) are skipped — used to
    guarantee no source-post overlap between successive generation rounds.
    """
    pos_kws = [k.lower() for k in EMOTION_REDDIT_KEYWORDS.get(target_pos, [])]
    neg_kws = [k.lower() for k in EMOTION_REDDIT_KEYWORDS.get(target_neg, [])]
    excluded = _load_excluded_post_ids()
    matches: list[dict] = []
    for line in lines:
        if len(matches) >= n_wanted:
            break
        try:
            post = json.loads(line)
        except Exception:
            continue
        if excluded and post.get("id") in excluded:
            continue
        text = (post.get("selftext") or "").lower()
        if not text:
            continue
        if any(k in text for k in pos_kws) or any(k in text for k in neg_kws):
            matches.append(post)
    return matches


# ── Prompt extension for forced targets ──────────────────────────────────────

def build_forced_render_prompt(base_prompt: str, target_pos: str, target_neg: str) -> str:
    """Append a MANDATORY emotions block after the base render prompt."""
    forced = (
        "\n\nMANDATORY EMOTIONS (non-negotiable):\n"
        f"  Agent A's emotion MUST be exactly: {target_pos}\n"
        f"  Agent B's emotion MUST be exactly: {target_neg}\n"
        "Do NOT return a different emotion. If the AITA inspiration does not\n"
        "cleanly fit these emotions, adapt the scene — invent a helper, a\n"
        "reveal, a public audience, a future event, whatever is needed — so\n"
        "that the required triggers for each emotion appear in the sentence.\n"
        f"  {target_pos}: "
        + {
            "joy":       "agent gains something good (outcome already happened, no prior threat).",
            "pride":     "agent built/earned/completed/passed something through their own effort.",
            "relief":    "a prior threat/worry is shown and then AVOIDED.",
            "gratitude": "another identifiable person specifically helps the agent.",
            "excitement":"something good is ABOUT to happen but has NOT happened yet.",
            "anger":     "another person treats the agent unfairly; a clear wrongdoer exists.",
            "sadness":   "agent loses something; no one is to blame (circumstance/bad luck).",
            "fear":      "a bad outcome has not yet happened but might (threat still active).",
            "disgust":   "someone does something morally REVOLTING; agent is repulsed by behavior.",
            "embarrassment":"agent is exposed/fails with other people watching (public audience).",
        }[target_pos]
        + f"\n  {target_neg}: "
        + {
            "joy":       "agent gains something good.",
            "pride":     "agent's own effort/skill produced the outcome.",
            "relief":    "prior threat avoided.",
            "gratitude": "helper is identifiable in the sentence.",
            "excitement":"future event not yet realized.",
            "anger":     "another person treats the agent unfairly.",
            "sadness":   "loss with no one to blame.",
            "fear":      "unresolved threat.",
            "disgust":   "someone's morally revolting behavior — the agent is repulsed.",
            "embarrassment":"public audience watches the exposure.",
        }[target_neg]
        + "\n"
    )
    return base_prompt + forced


# ── Render helpers (force target, retry on mismatch) ─────────────────────────

def render_with_target(draft, base_prompt, schema, label, target_pos, target_neg,
                       max_attempts: int = 4):
    prompt = build_forced_render_prompt(base_prompt, target_pos, target_neg)
    best_blob = None
    best_issues = None
    for attempt in range(max_attempts):
        blob = schema_call(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(draft, ensure_ascii=False)},
            ],
            schema=schema,
            max_tokens=5000,
        )

        issues = list(validate_version(blob, label))
        # Extra check: emotion must match target exactly
        if blob["agent_A"]["emotion"] != target_pos:
            issues.append(
                f"{label}: Agent A emotion must be '{target_pos}', got "
                f"'{blob['agent_A']['emotion']}'"
            )
        if blob["agent_B"]["emotion"] != target_neg:
            issues.append(
                f"{label}: Agent B emotion must be '{target_neg}', got "
                f"'{blob['agent_B']['emotion']}'"
            )

        if not issues:
            return blob
        if best_blob is None or len(issues) < len(best_issues):
            best_blob, best_issues = blob, issues
        print(f"  [{label}] attempt {attempt+1}/{max_attempts}: {issues}")

        # Repair with explicit emotion requirement reiterated
        repair_user = (
            "Problems:\n" + "\n".join(issues) +
            f"\n\nRequired emotions: Agent A = {target_pos}, Agent B = {target_neg}.\n"
            "You MUST NOT use any other emotion for either agent."
        )
        blob = schema_call(
            messages=[
                {"role": "system", "content": _REPAIR_SYSTEM},
                {"role": "user", "content": repair_user},
                {"role": "user", "content": "Current JSON:\n" + json.dumps(blob, ensure_ascii=False)},
            ],
            schema=schema,
            max_tokens=5000,
        )
        issues = list(validate_version(blob, label))
        if blob["agent_A"]["emotion"] != target_pos:
            issues.append("wrong A emotion after repair")
        if blob["agent_B"]["emotion"] != target_neg:
            issues.append("wrong B emotion after repair")
        if not issues:
            return blob
        if len(issues) < len(best_issues):
            best_blob, best_issues = blob, issues

    print(f"  [{label}] WARNING: final issues: {best_issues}")
    return best_blob


# ── Main ──────────────────────────────────────────────────────────────────────

_PROGRESS_LOCK = __import__("threading").Lock()
# Within-run set of post IDs already consumed. Threads claim a post under
# this lock to guarantee each Reddit post is used by at most one story per
# run (no in-run repetition).
_IN_RUN_USED_POSTS: set[str] = set()
_IN_RUN_LOCK = __import__("threading").Lock()


def _claim_unique_post(candidates: list[dict], rng_local: random.Random) -> dict | None:
    """Pick a post from `candidates` whose `id` has not been claimed yet
    by another thread in this run. Marks the chosen id as claimed."""
    pool = candidates[:]
    rng_local.shuffle(pool)
    with _IN_RUN_LOCK:
        for post in pool:
            pid = post.get("id")
            if pid and pid not in _IN_RUN_USED_POSTS:
                _IN_RUN_USED_POSTS.add(pid)
                return post
    return None


def _process_one(i: int, tp: str, tn: str, idx_base: int, lines: list[str],
                 existing: set[str], pos_dist: Counter, neg_dist: Counter,
                 done_counter: list[int]) -> str | None:
    story_idx = idx_base + i
    if str(story_idx) in existing:
        return None

    rng_local = random.Random(42 + i)
    candidates = filter_reddit_posts(lines, tp, tn, n_wanted=200)
    post = _claim_unique_post(candidates, rng_local)
    if post is None:
        # Fallback: broaden to any non-excluded post with selftext.
        excluded = _load_excluded_post_ids()
        broader = []
        for line in lines:
            try:
                p = json.loads(line)
            except Exception:
                continue
            if excluded and p.get("id") in excluded:
                continue
            if (p.get("selftext") or "").strip():
                broader.append(p)
        post = _claim_unique_post(broader, rng_local)
        if post is None:
            with _PROGRESS_LOCK:
                print(f"  [warn] no unique post available for idx={story_idx} pair=({tp},{tn})")
            return None
    story_text = (post.get("selftext") or "")[:2000]
    title = post.get("title", f"story_{story_idx}")

    category_phys = rng_local.choice(CONTRASTIVE_CATEGORIES)
    category_nonphys = rng_local.choice(CONTRASTIVE_CATEGORIES)

    with _PROGRESS_LOCK:
        done_counter[1] += 1
        print(f"\n[start {done_counter[1]}/{NUM_STORIES}] target=({tp}, {tn}) idx={story_idx}")
        print(f"  Source: {title[:60]}...")

    guidance = f"Pair MUST be Agent A={tp} (positive), Agent B={tn} (negative). Build the scene around this pair."
    draft_prompt = DRAFT_PROMPT_TEMPLATE.format(
        category_block=_build_category_block(category_phys),
        emotion_guidance=guidance,
    )
    draft_phys = schema_call(
        messages=[
            {"role": "system", "content": draft_prompt},
            {"role": "user", "content": f"Story inspiration:\n{story_text}"},
        ],
        schema=DRAFT_SCHEMA,
        max_tokens=5000,
        temperature=1.0,
    )

    from generate_stories_v2 import PHYS_RENDER_PROMPT, NONPHYS_RENDER_PROMPT
    phys = render_with_target(
        draft_phys, PHYS_RENDER_PROMPT, PHYS_SCHEMA, "physical", tp, tn,
    )

    draft_prompt_np = DRAFT_PROMPT_TEMPLATE.format(
        category_block=_build_category_block(category_nonphys),
        emotion_guidance=guidance,
    )
    draft_nonphys = schema_call(
        messages=[
            {"role": "system", "content": draft_prompt_np},
            {"role": "user", "content": f"Story inspiration:\n{story_text}"},
        ],
        schema=DRAFT_SCHEMA,
        max_tokens=5000,
        temperature=1.0,
    )
    nonphys = render_with_target(
        draft_nonphys, NONPHYS_RENDER_PROMPT, NONPHYS_SCHEMA, "non_physical", tp, tn,
    )

    with _PROGRESS_LOCK:
        pos_dist[phys["agent_A"]["emotion"]] += 1
        neg_dist[phys["agent_B"]["emotion"]] += 1
        pos_dist[nonphys["agent_A"]["emotion"]] += 1
        neg_dist[nonphys["agent_B"]["emotion"]] += 1

    final = {
        "category_physical": category_phys["name"],
        "category_non_physical": category_nonphys["name"],
        "setting_physical": draft_phys["setting"],
        "setting_non_physical": draft_nonphys["setting"],
        "physical_version": phys,
        "non_physical_version": nonphys,
        "_target_emotions": {"positive": tp, "negative": tn},
        "_source_post_id": post.get("id"),
        "_source_post_title": post.get("title"),
    }

    safe_setting = re.sub(r'[\\/:*?"<>|]', "-", draft_phys["setting"][:80])
    outpath = os.path.join(OUTPUT_DIR, f"story_{story_idx}_{safe_setting}.json")
    with open(outpath, "w", encoding="utf-8") as out:
        json.dump(final, out, indent=2, ensure_ascii=False)

    with _PROGRESS_LOCK:
        done_counter[0] += 1
        print(f"  [done {done_counter[0]}/{NUM_STORIES}] Saved: {os.path.basename(outpath)}")
        print(f"  Pos so far: {dict(pos_dist)} | Neg so far: {dict(neg_dist)}")
    return outpath


def main() -> None:
    with open(REDDIT_DUMP, "r", encoding="utf-8") as f:
        lines = f.readlines()

    existing = {
        os.path.basename(p).split("_")[1]
        for p in _glob(os.path.join(OUTPUT_DIR, "story_*.json"))
    }
    if "STORY_IDX_BASE" in os.environ:
        idx_base = int(os.environ["STORY_IDX_BASE"])
    elif existing:
        idx_base = max(int(x) for x in existing if x.isdigit()) + 1
    else:
        idx_base = 0

    # Durable in-run uniqueness: pre-seed _IN_RUN_USED_POSTS with every
    # _source_post_id already present on disk. This way a restart cannot
    # re-pick a post that an earlier (now-dead) process already used.
    pre_seeded = 0
    for p in _glob(os.path.join(OUTPUT_DIR, "story_*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        pid = d.get("_source_post_id")
        if pid:
            _IN_RUN_USED_POSTS.add(pid)
            pre_seeded += 1
    if pre_seeded:
        print(f"[in-run-uniqueness] pre-seeded {pre_seeded} post ids from existing stories in {OUTPUT_DIR}/")

    pos_dist: Counter = Counter()
    neg_dist: Counter = Counter()
    done_counter = [0, 0]  # [completed, started]

    if WORKERS <= 1:
        for i, (tp, tn) in enumerate(TARGET_PAIRS):
            _process_one(i, tp, tn, idx_base, lines, existing,
                         pos_dist, neg_dist, done_counter)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"[parallel] running {WORKERS} workers")
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = [
                ex.submit(_process_one, i, tp, tn, idx_base, lines,
                          existing, pos_dist, neg_dist, done_counter)
                for i, (tp, tn) in enumerate(TARGET_PAIRS)
            ]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  [worker error] {e}")

    print(f"\n[OK] Generated {NUM_STORIES} balanced stories -> {OUTPUT_DIR}/")
    print(f"  Positive distribution: {dict(pos_dist)}")
    print(f"  Negative distribution: {dict(neg_dist)}")


if __name__ == "__main__":
    main()
