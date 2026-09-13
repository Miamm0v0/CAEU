"""
generate_mcq.py

Generates multiple-choice questions (MCQs) for contrastive emotion evaluation.
Reads story JSON files from AITA_no_visual and/or AITA_no_visual_diverse_grain.
No LLM calls needed — all generation is deterministic.

Output:
  mcq_dataset.json  — flat list of all MCQ instances for eval_mcq.py

MCQ types generated per story version:
  MCQ-2a: single_agent_A     — which emotion does Agent A feel? (5 same-polarity options)
  MCQ-2b: single_agent_B     — which emotion does Agent B feel? (5 same-polarity options)
  MCQ-3a: causal_A           — which text span explains Agent A's emotion?
  MCQ-3b: causal_B           — which text span explains Agent B's emotion?
  MCQ-4a: emotion_evidence_A — combined: Agent A feels ___ because ___
  MCQ-4b: emotion_evidence_B — combined: Agent B feels ___ because ___

Distractor design:
  MCQ-2 options (5-way, all same polarity):
    All 5 emotions from the correct polarity bucket.
    No elimination possible — forces genuine fine-grained discrimination.

  MCQ-3 options (4-way):
    Correct:      evidence for the target agent
    Distractor 1: evidence for the other agent   (tests agent attribution)
    Distractor 2: cause_span                     (cause, not consequence)
    Distractor 3: evidence for target agent from the OTHER version (physical <-> non_physical)
                  Falls back to setting fragment if cross-version span unavailable.

  MCQ-4 options (4-way, combined emotion + evidence):
    Correct:      correct_emotion — correct_evidence
    Distractor 1: wrong_emotion  — correct_evidence   (emotion error)
    Distractor 2: correct_emotion — wrong_evidence     (evidence error)
    Distractor 3: wrong_emotion  — wrong_evidence      (both wrong)

Run:
  python generate_mcq.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# CONFIG
# =============================================================================

DATASET_DIRS = os.getenv("DATASET_DIRS", "stories_v2").split(":")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "mcq_dataset_v2.json")

# =============================================================================
# EMOTION CONSTANTS (v2 taxonomy)
# =============================================================================

POS_EMOTIONS = ["joy", "pride", "relief", "gratitude", "excitement"]
NEG_EMOTIONS = ["anger", "sadness", "fear", "disgust", "embarrassment"]
ALL_EMOTIONS = POS_EMOTIONS + NEG_EMOTIONS
EMOTION_SET  = set(ALL_EMOTIONS)

# Same-polarity alternatives (4 per emotion).
SAME_POLARITY_POOL: Dict[str, List[str]] = {
    emo: [e for e in POS_EMOTIONS if e != emo]
    for emo in POS_EMOTIONS
}
SAME_POLARITY_POOL.update({
    emo: [e for e in NEG_EMOTIONS if e != emo]
    for emo in NEG_EMOTIONS
})

# Opposite-polarity pool for each emotion.
OPP_POLARITY_POOL: Dict[str, List[str]] = {
    emo: list(NEG_EMOTIONS) for emo in POS_EMOTIONS
}
OPP_POLARITY_POOL.update({
    emo: list(POS_EMOTIONS) for emo in NEG_EMOTIONS
})

# =============================================================================
# UTILITIES
# =============================================================================

def safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def shuffle_options(
    options: List[str],
    rotation: int,
) -> Tuple[Dict[str, str], str]:
    """
    Place N options (4 or 5) into letter slots such that the first element
    (the correct answer) lands at position (rotation % N).

    Args:
        options: [correct, distractor1, distractor2, ...] — 4 or 5 items
        rotation: deterministic counter to rotate correct-answer position

    Returns:
        (options_dict, correct_letter)
    """
    n = len(options)
    letters = ["A", "B", "C", "D", "E"][:n]
    target_pos = rotation % n

    ordered: List[str] = [""] * n
    ordered[target_pos] = options[0]  # correct answer

    other = options[1:]  # distractors
    remaining = [i for i in range(n) if i != target_pos]
    for i, pos in enumerate(remaining):
        ordered[pos] = other[i]

    options_dict = {letters[i]: ordered[i] for i in range(n)}
    correct_letter = letters[target_pos]
    return options_dict, correct_letter


def truncate_setting(setting: str, max_chars: int = 80) -> str:
    """Extract a short landmark phrase from the setting for use as a distractor."""
    s = (setting or "").strip()
    if not s:
        return "the surrounding area"
    for sep in [",", ".", " with ", " near ", " and "]:
        idx = s.find(sep)
        if 0 < idx < max_chars:
            return s[:idx].strip()
    return s[:max_chars].strip()


# =============================================================================
# MCQ GENERATORS
# =============================================================================

def make_mcq2_single_agent(
    correct_emotion: str,
    agent_label: str,
    role_a: str,
    role_b: str,
    rotation: int,
) -> Optional[Dict[str, Any]]:
    """
    MCQ-2: Single-agent emotion (5-way, all same polarity).
    All 5 emotions from the correct polarity bucket are shown —
    no elimination possible, forces genuine fine-grained discrimination.

    Args:
        role_a: Always Agent A's role (regardless of which agent is the target).
        role_b: Always Agent B's role (regardless of which agent is the target).
    """
    if correct_emotion not in EMOTION_SET:
        return None

    same_pol = SAME_POLARITY_POOL[correct_emotion]  # list of 4
    all_options = [correct_emotion] + same_pol       # 5 total

    options_dict, correct_letter = shuffle_options(all_options, rotation)

    return {
        "question": (
            f"The sentence describes two people reacting to the same event with "
            f"contrasting emotions.\n\n"
            f"Agent A is the {role_a}; Agent B is the {role_b}.\n\n"
            f"What emotion does Agent {agent_label} feel?"
        ),
        "options": options_dict,
        "correct": correct_letter,
    }


def make_mcq3_causal_attr(
    correct_evidence: str,
    other_evidence: str,
    cause_span: str,
    cross_version_evidence: str,
    setting: str,
    agent_label: str,
    agent_role: str,
    emotion: str,
    rotation: int,
) -> Optional[Dict[str, Any]]:
    """
    MCQ-3: Causal attribution (4-way, harder distractors).

    Options:
      Correct:      evidence for target agent
      Distractor 1: evidence for the other agent   (wrong agent)
      Distractor 2: cause_span                     (cause, not consequence)
      Distractor 3: cross-version evidence for same agent (from physical<->non_physical)
                    Falls back to setting fragment if unavailable or duplicate.
    """
    if not correct_evidence or not correct_evidence.strip():
        return None

    setting_span = truncate_setting(setting)

    # Build distractors in priority order
    seen: set = {correct_evidence.strip()}
    dists: List[str] = []

    # Distractor 1: other agent's evidence
    c = (other_evidence or "").strip()
    if c and c not in seen:
        seen.add(c)
        dists.append(c)

    # Distractor 2: cause span (too general)
    c = (cause_span or "").strip()
    if c and c not in seen:
        seen.add(c)
        dists.append(c)

    # Distractor 3: cross-version evidence (harder — same agent, different version)
    c = (cross_version_evidence or "").strip()
    if c and c not in seen:
        seen.add(c)
        dists.append(c)
    else:
        # Fallback: setting fragment
        c = setting_span
        if c and c not in seen:
            seen.add(c)
            dists.append(c)

    # Fill any remaining gaps
    while len(dists) < 3:
        fallback = setting_span[:50] if len(setting_span) > 50 else (setting_span + " area")
        if fallback not in seen:
            seen.add(fallback)
            dists.append(fallback)
        else:
            break

    if len(dists) < 3:
        return None  # can't build 4 distinct options

    options_dict, correct_letter = shuffle_options(
        [correct_evidence.strip()] + dists[:3], rotation
    )

    return {
        "question": (
            f"Which text span from the sentence most directly explains "
            f"why Agent {agent_label} (the {agent_role}) feels {emotion}?"
        ),
        "options": options_dict,
        "correct": correct_letter,
    }


def make_mcq4_emotion_evidence(
    correct_emotion: str,
    correct_evidence: str,
    other_evidence: str,
    agent_label: str,
    agent_role: str,
    rotation: int,
) -> Optional[Dict[str, Any]]:
    """
    MCQ-4: Combined emotion + evidence (4-way).
    Forces joint reasoning — the right emotion must match the right evidence.

    Options:
      Correct:      correct_emotion — correct_evidence
      Distractor 1: wrong_emotion  — correct_evidence   (emotion error only)
      Distractor 2: correct_emotion — wrong_evidence     (evidence error only)
      Distractor 3: wrong_emotion  — wrong_evidence      (both wrong)
    """
    if correct_emotion not in EMOTION_SET:
        return None
    if not correct_evidence or not correct_evidence.strip():
        return None
    if not other_evidence or not other_evidence.strip():
        return None

    same_pol = SAME_POLARITY_POOL[correct_emotion]
    wrong_emotion = same_pol[0]  # first same-polarity alternative
    wrong_evidence = other_evidence.strip()
    correct_ev = correct_evidence.strip()

    correct_str = f"{correct_emotion} — {correct_ev}"
    dist1_str   = f"{wrong_emotion} — {correct_ev}"
    dist2_str   = f"{correct_emotion} — {wrong_evidence}"
    dist3_str   = f"{wrong_emotion} — {wrong_evidence}"

    # Ensure all four are distinct
    if len({correct_str, dist1_str, dist2_str, dist3_str}) < 4:
        return None

    options_dict, correct_letter = shuffle_options(
        [correct_str, dist1_str, dist2_str, dist3_str], rotation
    )

    return {
        "question": (
            f"The sentence describes two people reacting to the same event.\n\n"
            f"Agent {agent_label} (the {agent_role}) feels _____ because _____.\n\n"
            f"Which emotion-and-evidence combination is correct?"
        ),
        "options": options_dict,
        "correct": correct_letter,
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_story_files(base_dir: str) -> List[Dict[str, Any]]:
    """Load all story_*.json files from a directory."""
    pattern = os.path.join(base_dir, "story_*.json")
    stories: List[Dict[str, Any]] = []
    for fpath in sorted(glob(pattern)):
        try:
            obj = safe_load_json(fpath)
            obj["_file"] = fpath
            stories.append(obj)
        except Exception as e:
            print(f"  [SKIP] {fpath}: {e}")
    return stories


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    all_mcqs: List[Dict[str, Any]] = []
    mcq_counter = 0
    story_version_index = 0  # odd = swap Agent A/B polarity

    for base_dir in DATASET_DIRS:
        if not os.path.isdir(base_dir):
            print(f"[SKIP] Directory not found: {base_dir}")
            continue

        stories = load_story_files(base_dir)
        print(f"[INFO] {base_dir}: loaded {len(stories)} story files")

        for story in stories:
            file_path = story["_file"]

            # Pre-extract both versions so we can cross-reference evidence spans
            versions: Dict[str, Dict[str, Any]] = {}
            for ver_key, ver_label in [
                ("physical_version",     "physical"),
                ("non_physical_version", "non_physical"),
            ]:
                blob = story.get(ver_key)
                if isinstance(blob, dict):
                    versions[ver_label] = blob

            for ver_label, blob in versions.items():
                # Pick the setting appropriate to this version
                if ver_label == "physical" and "setting_physical" in story:
                    setting = story.get("setting_physical", "") or ""
                elif ver_label == "non_physical" and "setting_non_physical" in story:
                    setting = story.get("setting_non_physical", "") or ""
                else:
                    setting = story.get("setting", "") or ""

                a          = blob.get("agent_A") or {}
                b          = blob.get("agent_B") or {}
                sentence   = (blob.get("sentence")   or "").strip()
                cause_span = (blob.get("cause_span")  or "").strip()
                evidence_a = (blob.get("evidence_A")  or "").strip()
                evidence_b = (blob.get("evidence_B")  or "").strip()
                emo_a      = (a.get("emotion") or "").strip().lower()
                emo_b      = (b.get("emotion") or "").strip().lower()
                role_a     = (a.get("role")    or "Agent A").strip()
                role_b     = (b.get("role")    or "Agent B").strip()

                if not sentence:
                    continue

                # Cross-version evidence for harder causal distractors
                other_ver = "non_physical" if ver_label == "physical" else "physical"
                cross_blob = versions.get(other_ver, {})
                cross_evidence_a = (cross_blob.get("evidence_A") or "").strip()
                cross_evidence_b = (cross_blob.get("evidence_B") or "").strip()

                # ── Polarity swap: for half of story+version pairs, swap A/B ──
                swap = (story_version_index % 2 == 1)
                story_version_index += 1

                if swap:
                    emo_a, emo_b = emo_b, emo_a
                    role_a, role_b = role_b, role_a
                    evidence_a, evidence_b = evidence_b, evidence_a
                    cross_evidence_a, cross_evidence_b = cross_evidence_b, cross_evidence_a

                agent_a_polarity = "negative" if swap else "positive"
                agent_b_polarity = "positive" if swap else "negative"

                base_id = f"{os.path.basename(file_path).replace('.json', '')}_{ver_label}"
                dataset_name = os.path.basename(base_dir)

                shared_fields = {
                    "dataset":          dataset_name,
                    "story_file":       file_path,
                    "version":          ver_label,
                    "sentence":         sentence,
                    "agent_a_role":     role_a,
                    "agent_b_role":     role_b,
                    "gold_emotion_A":   emo_a,
                    "gold_emotion_B":   emo_b,
                    "agent_a_polarity": agent_a_polarity,
                    "agent_b_polarity": agent_b_polarity,
                    "swapped":          swap,
                }

                # ── MCQ-2a: Agent A emotion (5-way, same-polarity) ────────────
                mcq2a = make_mcq2_single_agent(
                    emo_a, "A", role_a=role_a, role_b=role_b, rotation=mcq_counter
                )
                if mcq2a:
                    all_mcqs.append({
                        "mcq_id":   f"{base_id}_mcq2a",
                        "mcq_type": "single_agent_A",
                        **shared_fields,
                        **mcq2a,
                    })
                    mcq_counter += 1

                # ── MCQ-2b: Agent B emotion (5-way, same-polarity) ────────────
                mcq2b = make_mcq2_single_agent(
                    emo_b, "B", role_a=role_a, role_b=role_b, rotation=mcq_counter
                )
                if mcq2b:
                    all_mcqs.append({
                        "mcq_id":   f"{base_id}_mcq2b",
                        "mcq_type": "single_agent_B",
                        **shared_fields,
                        **mcq2b,
                    })
                    mcq_counter += 1

                # ── MCQ-3a: Causal attribution for Agent A ────────────────────
                if evidence_a and cause_span:
                    mcq3a = make_mcq3_causal_attr(
                        evidence_a, evidence_b, cause_span,
                        cross_version_evidence=cross_evidence_a,
                        setting=setting,
                        agent_label="A", agent_role=role_a, emotion=emo_a,
                        rotation=mcq_counter,
                    )
                    if mcq3a:
                        all_mcqs.append({
                            "mcq_id":          f"{base_id}_mcq3a",
                            "mcq_type":        "causal_A",
                            "gold_cause_span": cause_span,
                            "gold_evidence_A": evidence_a,
                            "gold_evidence_B": evidence_b,
                            **shared_fields,
                            **mcq3a,
                        })
                        mcq_counter += 1

                # ── MCQ-3b: Causal attribution for Agent B ────────────────────
                if evidence_b and cause_span:
                    mcq3b = make_mcq3_causal_attr(
                        evidence_b, evidence_a, cause_span,
                        cross_version_evidence=cross_evidence_b,
                        setting=setting,
                        agent_label="B", agent_role=role_b, emotion=emo_b,
                        rotation=mcq_counter,
                    )
                    if mcq3b:
                        all_mcqs.append({
                            "mcq_id":          f"{base_id}_mcq3b",
                            "mcq_type":        "causal_B",
                            "gold_cause_span": cause_span,
                            "gold_evidence_A": evidence_a,
                            "gold_evidence_B": evidence_b,
                            **shared_fields,
                            **mcq3b,
                        })
                        mcq_counter += 1

                # ── MCQ-4a: Emotion + evidence for Agent A ────────────────────
                if evidence_a and evidence_b:
                    mcq4a = make_mcq4_emotion_evidence(
                        emo_a, evidence_a, evidence_b,
                        agent_label="A", agent_role=role_a,
                        rotation=mcq_counter,
                    )
                    if mcq4a:
                        all_mcqs.append({
                            "mcq_id":   f"{base_id}_mcq4a",
                            "mcq_type": "emotion_evidence_A",
                            "gold_evidence_A": evidence_a,
                            "gold_evidence_B": evidence_b,
                            **shared_fields,
                            **mcq4a,
                        })
                        mcq_counter += 1

                # ── MCQ-4b: Emotion + evidence for Agent B ────────────────────
                if evidence_a and evidence_b:
                    mcq4b = make_mcq4_emotion_evidence(
                        emo_b, evidence_b, evidence_a,
                        agent_label="B", agent_role=role_b,
                        rotation=mcq_counter,
                    )
                    if mcq4b:
                        all_mcqs.append({
                            "mcq_id":   f"{base_id}_mcq4b",
                            "mcq_type": "emotion_evidence_B",
                            "gold_evidence_A": evidence_a,
                            "gold_evidence_B": evidence_b,
                            **shared_fields,
                            **mcq4b,
                        })
                        mcq_counter += 1

    # ── Write output ──────────────────────────────────────────────────────────
    atomic_write_json(OUTPUT_FILE, all_mcqs)

    print(f"\n[OK] Generated {len(all_mcqs)} MCQs -> {OUTPUT_FILE}")
    print(f"  By type:    {dict(Counter(m['mcq_type'] for m in all_mcqs))}")
    print(f"  By version: {dict(Counter(m['version']  for m in all_mcqs))}")
    print(f"  By dataset: {dict(Counter(m['dataset']  for m in all_mcqs))}")
    print(f"  By swap:    {dict(Counter(m['swapped']  for m in all_mcqs))}")

    # Print a sample MCQ of each type for verification
    print("\n-- Sample MCQs ------------------------------------------------------------------")
    seen_types: set = set()
    for mcq in all_mcqs:
        if mcq["mcq_type"] not in seen_types:
            seen_types.add(mcq["mcq_type"])
            print(f"\n[{mcq['mcq_type']}]")
            print(f"  Sentence: {mcq['sentence'][:100]}...")
            print(f"  Question: {mcq['question']}")
            for k, v in sorted(mcq["options"].items()):
                marker = " <-- CORRECT" if k == mcq["correct"] else ""
                print(f"    {k}. {v}{marker}")
        if len(seen_types) == 6:
            break


if __name__ == "__main__":
    main()
