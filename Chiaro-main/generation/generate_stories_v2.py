"""
generate_stories_v2.py

Generates 25 contrastive emotion stories using the v2 taxonomy.
Adapted from exp_no_visual_separate_calls.py.

v2 taxonomy:
  Positive: joy, pride, relief, gratitude, excitement
  Negative: anger, sadness, fear, disgust, embarrassment

Pipeline per story:
  1. Pick contrastive category + target emotions (for balance)
  2. Draft scene from AITA Reddit post
  3. Render physical version
  4. Render non-physical version
  5. Validate & repair
  6. Save to stories_v2/

Run:
  python generate_stories_v2.py
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import os
import random
import re
import time
from collections import Counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL", "gpt-5.2")
TEMPERATURE = 0.8
NUM_STORIES = 75
REDDIT_OFFSET = 160  # Skip posts already used in 3-25 (60-159)

OUTPUT_DIR = "stories_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REDDIT_DUMP = os.path.join("..", "reddit", "AmItheAsshole_reddit_dump.jsonl")

# ── Emotion taxonomy (v2) ────────────────────────────────────────────────────

POSITIVE = {"joy", "pride", "relief", "gratitude", "excitement"}
NEGATIVE = {"anger", "sadness", "fear", "disgust", "embarrassment"}
ALL_EMOTIONS = POSITIVE | NEGATIVE


def has_contrast(a, b):
    return (a in POSITIVE and b in NEGATIVE) or (b in POSITIVE and a in NEGATIVE)


# ── Banned words ──────────────────────────────────────────────────────────────

BANNED = {
    "happy", "happily", "joy", "joyful", "delighted",
    "proud", "pride", "prideful",
    "relieved", "relief",
    "grateful", "gratitude", "thankful",
    "excited", "excitement", "eager", "eagerly",
    "sad", "sadly", "sorrow", "sorrowful",
    "angry", "anger", "furious", "mad", "enraged",
    "fear", "fearful", "afraid", "scared", "terrified",
    "guilty", "guilt", "regret", "regretful",
    "disgust", "disgusted", "disgusting", "revolting", "repulsed",
    "embarrassed", "embarrassment", "ashamed", "humiliated", "shame",
    "upset", "annoyed", "frustrated", "dismayed",
    "slumped shoulders", "tight jaw", "teary eyes", "welling eyes",
    "visibly", "clearly", "nervously", "nods", "smiles", "frowns",
    "glares", "cries", "screams", "laughs", "grins", "claps",
    "snatches", "pumps", "yanks", "raises a fist", "jumps for joy",
    "throws up his hands",
}
BANNED_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, sorted(BANNED))) + r")\b",
    re.IGNORECASE,
)


def contains_banned(text: str | None) -> bool:
    return BANNED_RE.search(text or "") is not None


# ── Emotion triggers (mandatory story elements for disambiguation) ────────────

EMOTION_TRIGGERS = {
    "joy":           "The agent receives or gains something good — the positive outcome has ALREADY happened.",
    "pride":         "The agent accomplished something through their OWN effort, skill, or hard work. The sentence must show personal achievement.",
    "relief":        "The sentence must show a PRIOR THREAT or WORRY that was then AVOIDED or resolved. Without the threat, it's just joy.",
    "gratitude":     "ANOTHER PERSON specifically helped, supported, or sacrificed for the agent. The helper must be identifiable.",
    "excitement":    "Something good is ABOUT TO happen but hasn't yet. The agent is looking forward to a FUTURE event.",
    "anger":         "ANOTHER PERSON treated the agent unfairly, unjustly, or selfishly. There must be a clear wrongdoer.",
    "sadness":       "The agent lost something or was denied something, but there is NO ONE to blame — it's circumstance, bad luck, or an impersonal outcome.",
    "fear":          "A BAD OUTCOME has NOT happened yet but MIGHT. The threat is still active and unresolved.",
    "disgust":       "Someone did something morally revolting or repulsive. The agent is repulsed by another person's behavior, not necessarily victimized.",
    "embarrassment": "The agent was exposed, shamed, or failed with OTHER PEOPLE visibly WATCHING. A public audience must be present.",
}


# ── Contrastive categories ────────────────────────────────────────────────────

CONTRASTIVE_CATEGORIES = [
    {
        "name": "zero_sum_gain_loss",
        "description": "ZERO-SUM GAIN / LOSS — one agent GAINS something and the other LOSES or is DENIED that same thing.",
        "examples": [
            "One person gets the last ticket; the other arrives at an empty counter.",
            "One student sees an A on their paper; the other sees an F on theirs.",
        ],
    },
    {
        "name": "side_effect_spillover",
        "description": "SIDE-EFFECT / SPILLOVER — Agent A's activity makes them feel positive, but a byproduct causes a negative experience for Agent B.",
        "examples": [
            "A child bounces in their airplane seat from excitement and keeps kicking the seat-back, bothering the passenger in front.",
            "A musician practices a new song in their apartment while the neighbor can't concentrate.",
        ],
    },
    {
        "name": "asymmetric_information",
        "description": "ASYMMETRIC INFORMATION — The same event is experienced differently because agents have different knowledge or stakes.",
        "examples": [
            "A student learns they got early admission while their friend hasn't heard back.",
            "A worker finds out they passed probation while the colleague's contract won't be renewed.",
        ],
    },
    {
        "name": "unintended_consequence",
        "description": "UNINTENDED CONSEQUENCE — Agent A acts with a positive purpose, but an unintended side-effect harms Agent B.",
        "examples": [
            "A gardener waters flowers and the runoff floods the neighbor's mulch.",
            "A teacher rearranges seating for a reading corner, but one student loses their window seat.",
        ],
    },
    {
        "name": "competing_preferences",
        "description": "COMPETING PREFERENCES — Two agents share an environment but have opposing needs. Satisfying one automatically works against the other.",
        "examples": [
            "A parent turns on the AC but their child was already cold.",
            "One roommate opens the window for a breeze while the other's papers blow off the desk.",
        ],
    },
    {
        "name": "success_vs_failure",
        "description": "SUCCESS vs. FAILURE — Both agents attempt the same challenge. One succeeds and the other fails independently.",
        "examples": [
            "One runner finishes the marathon while another drops out from a cramp.",
            "One baker's soufflé rises perfectly while the other's collapses.",
        ],
    },
]

# ── JSON Schemas ──────────────────────────────────────────────────────────────

DRAFT_SCHEMA = {
    "name": "draft_scene",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "setting":      {"type": "string"},
            "agent_A_role": {"type": "string"},
            "agent_B_role": {"type": "string"},
            "draft_story":  {"type": "string"},
        },
        "required": ["setting", "agent_A_role", "agent_B_role", "draft_story"],
    },
}

_VERSION_BODY = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "agent_A": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "role":    {"type": "string"},
                "emotion": {"type": "string", "enum": sorted(ALL_EMOTIONS)},
                "action":  {"type": "string"},
            },
            "required": ["role", "emotion", "action"],
        },
        "agent_B": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "role":     {"type": "string"},
                "emotion":  {"type": "string", "enum": sorted(ALL_EMOTIONS)},
                "reaction": {"type": "string"},
            },
            "required": ["role", "emotion", "reaction"],
        },
        "cause_effect_relation": {"type": "string"},
        "sentence":   {"type": "string"},
        "cause_span": {"type": "string"},
        "evidence_A": {"type": "string"},
        "evidence_B": {"type": "string"},
    },
    "required": [
        "agent_A", "agent_B", "cause_effect_relation",
        "sentence", "cause_span", "evidence_A", "evidence_B",
    ],
}

PHYS_SCHEMA = {"name": "physical_version", "strict": True, "schema": _VERSION_BODY}
NONPHYS_SCHEMA = {"name": "non_physical_version", "strict": True, "schema": _VERSION_BODY}

# ── Prompts ───────────────────────────────────────────────────────────────────

DRAFT_PROMPT_TEMPLATE = """
You are writing a short draft scene involving exactly two human agents.

Constraints:
- Avoid stealing or creating villains.
- The emotions of the two agents are OPPOSITE in valence (one positive, one negative).
- Keep the language simple.
- The sentence must make clear WHY each agent feels the way they do.

EMOTION TAXONOMY — each emotion has a MANDATORY TRIGGER that MUST appear in the story:
  Positive:
    joy — Agent gains something good (outcome already happened)
    pride — Agent accomplished it through their OWN effort/skill (personal achievement must be shown)
    relief — A PRIOR THREAT or WORRY was AVOIDED (the threat must be shown, not just a good outcome)
    gratitude — ANOTHER PERSON specifically helped the agent (the helper must be identifiable)
    excitement — Something good is ABOUT TO happen but hasn't yet (future-oriented)
  Negative:
    anger — ANOTHER PERSON treated the agent unfairly (there must be a clear wrongdoer)
    sadness — Agent lost something but there is NO ONE TO BLAME (circumstance or bad luck)
    fear — A BAD OUTCOME has NOT happened yet but MIGHT (threat still active)
    disgust — Someone did something morally revolting (agent is repulsed by their behavior)
    embarrassment — Agent was exposed/failed with OTHER PEOPLE WATCHING (public audience required)

{emotion_guidance}

CONTRASTIVE SCENARIO TYPE:
{category_block}

Return JSON with: setting, agent_A_role, agent_B_role, draft_story
"""

EMOTION_DEFS = """
EMOTION TAXONOMY (use ONLY these):
  Positive: joy (happy about outcome), pride (accomplished something), relief (bad thing avoided), gratitude (thankful for help), excitement (eager about what's next)
  Negative: anger (unfair treatment), sadness (lost something), fear (something bad might happen), disgust (morally revolted), embarrassment (exposed/shamed publicly)
"""

_RENDER_CORE = """
SINGLE CAUSE RULE (most important):
- A SINGLE event must cause BOTH agents' emotions. Agent A's positive
  feeling and Agent B's negative feeling are both reactions to the SAME
  thing happening.

  VALID: "The last ticket is given to A (joy) — so B (sadness) cannot buy
          one." A single event (ticket handed over) causes both outcomes.
  INVALID: "A practices for her show tonight (excitement), while B sees
            trash in the hallway (disgust)." Two unrelated events.

- The cause_span must be a phrase in the sentence that names the SHARED
  event. Both evidence_A and evidence_B must be CONSEQUENCES of that
  shared event — never two different triggers.
- The cause_effect_relation must describe ONE cause producing two
  outcomes. Do NOT write "A's emotion comes from X, while B's emotion
  comes from Y" — that is two causes.

NEUTRAL CAUSE RULE (prevents adversarial framings):
- The shared event must be a NEUTRAL external happening or third-party
  action whose fallout affects A and B differently. The cause must NOT
  be Agent A deliberately acting against Agent B to hurt, punish,
  dominate, or extract payment from them.
- If A is directly targeting B (making them pay, reporting them, taking
  something from them, forcing them), the sentence reads as adversarial
  and A's "positive" feeling becomes punitive/vindictive, not genuinely
  positive. A reader will perceive BOTH emotions as negative.

  VALID (neutral event, outcomes split):
    - "The last ticket is handed to A, so B is turned away."
      (cause = clerk's neutral action)
    - "The school announces a single scholarship recipient."
      (cause = external announcement)
    - "A package meant for both arrives at A's desk first."
      (cause = the delivery, not A's choice)

  INVALID (A punishes / takes from B):
    - "A makes B pay for the damage."
      (A is actively coercing B; A's 'pride' is retaliatory)
    - "A reports B to the manager and gets a promotion."
      (A's gain comes from A's hostile action toward B)
    - "A takes the last slice from B's plate."
      (zero-sum, but A is the aggressor — reads as A wronging B)
  If you can only produce this pattern, pick a different contrastive
  category or rewrite the scene so the cause is external.

REQUIRED EMOTION CONTRAST:
- EXACTLY one POSITIVE and one NEGATIVE emotion per version.
- Agent A must have a POSITIVE emotion, Agent B must have a NEGATIVE emotion.
- Emotions must come from: positive={joy, pride, relief, gratitude, excitement}, negative={anger, sadness, fear, disgust, embarrassment}

CRITICAL — EMOTION DISAMBIGUATION RULES:
Each emotion has a MANDATORY TRIGGER. The sentence MUST contain this trigger or the emotion is WRONG:

  POSITIVE:
    joy — Agent receives/gains something good. Outcome ALREADY happened. No prior threat shown.
    pride — Agent accomplished something CONSTRUCTIVE through their own effort/skill (built, created, earned, passed, completed). Pride must NOT come from dominating, punishing, defeating, or extracting payment from Agent B — if the "accomplishment" is hurting/coercing B, this is not pride.
    relief — Sentence MUST show a PRIOR THREAT or WORRY that was then AVOIDED. Without showing the threat, use joy instead.
    gratitude — ANOTHER PERSON specifically helped the agent. The helper must be identifiable in the sentence.
    excitement — Something good is ABOUT TO happen but HASN'T YET. Agent is looking forward. If it already happened, use joy.

  NEGATIVE:
    anger — ANOTHER PERSON treated the agent unfairly/selfishly. A clear wrongdoer must exist.
    sadness — Agent lost something but NO ONE IS TO BLAME. It's circumstance or bad luck. If someone caused it, use anger.
    fear — Bad outcome has NOT happened yet but MIGHT. Threat is still active/unresolved. If it already happened, use sadness.
    disgust — Someone did something morally REVOLTING or repulsive. The agent is repulsed by another person's behavior. Different from anger (anger = unfair to YOU, disgust = morally wrong in general).
    embarrassment — Agent was exposed/failed with OTHER PEOPLE WATCHING. A public audience must be visible in the sentence.

SELF-CHECK after writing:
  - Does a SINGLE event explain both emotions? If A's cause and B's cause don't share a trigger, rewrite so they do.
  - Is Agent A's action directly targeting Agent B (punishing, coercing, taking from, reporting)? If yes, the sentence reads as adversarial — rewrite with a neutral external cause (a third party's decision, an announcement, an arrival, a rule that splits outcomes).
  - Read the sentence as if you didn't know which emotion was "positive". If BOTH feelings could be mistaken for negative (e.g., "A wins by punishing B"), rewrite so A's positive outcome is clearly constructive, not retaliatory.
  - If you used "relief": does the sentence show a threat that was avoided? If not, switch to "joy".
  - If you used "sadness": is there truly no one to blame? If someone caused it, switch to "anger".
  - If you used "fear": is the bad outcome still in the future? If it already happened, switch to "sadness".
  - If you used "embarrassment": are other people watching? If not, switch to "sadness" or "disgust".
  - If you used "excitement": is the good thing still in the future? If it already happened, switch to "joy".

HOW TO CONVEY CONTRAST WITHOUT EMOTION WORDS:
- The emotional contrast comes from WHAT HAPPENS — not from reactions or body language.
- Each agent's reason must be EXPLICITLY shown in the sentence.
- Do NOT use emotion words, facial expressions, or behavioral cues.

MOTIVATION RULE:
- The sentence must be SELF-CONTAINED. A reader with ZERO context must understand:
  1. WHAT happened
  2. WHY the positive agent feels positively (with the specific trigger for their emotion)
  3. WHY the negative agent feels negatively (with the specific trigger for their emotion)

AGENT ROLE RULE:
- Agent A and Agent B must have DIFFERENT role descriptions.
- The roles should clearly distinguish who is who.
- The role description must reference the agent USING THE SAME IDENTIFIER
  that appears in the sentence. If the sentence names a character
  ("Maya", "Alex"), use that name in the role. Do NOT introduce a
  relationship label ("Fiancé", "Roommate", "Cousin") that does not
  appear in the sentence — the annotator has no way to connect it to a
  named character.

  GOOD: sentence uses "Maya tapes a sign…Al puts up the same sign…"
        role_A: "Maya, who taped the first privacy sign"
        role_B: "Al, who copied Maya's sign and was questioned by his mother"

  BAD:  sentence uses names (Maya, Al) but role says "Fiancé trying to…"
        (Forces the reader to infer Fiancé == Al from outside context.)

- If the sentence does NOT use character names, describe each agent by
  their outcome or distinguishing trait visible in the sentence
  ("the trainee whose knot slips", "the shopper who reaches the counter
  first").

NATURAL LANGUAGE RULES:
- Present tense. Self-contained.
- No emotion words or behavioral cues in sentence.
- No dialogue or text on screens.

SPAN FIELDS:
- cause_span: exact phrase from sentence encoding the causing event.
- evidence_A: excerpt showing Agent A's situational outcome.
- evidence_B: excerpt showing Agent B's situational outcome.
"""

PHYS_RENDER_PROMPT = (
    "Convert a DRAFT scene into the PHYSICAL VERSION only.\n\n"
    "VERSION: physical_version — a concrete physical action or object change "
    "is the shared cause (e.g., last item grabbed from a shelf, door closed, "
    "pan pulled from oven, key handed over). The action need NOT be Agent A "
    "acting against Agent B — a neutral party, a mechanism, or even Agent B "
    "can be the one performing it. What matters is that the physical event "
    "produces different outcomes for both agents.\n\n" + _RENDER_CORE
)

NONPHYS_RENDER_PROMPT = (
    "Convert a DRAFT scene into the NON-PHYSICAL VERSION only.\n\n"
    "VERSION: non_physical_version — The cause is a situational/contextual cue, "
    "not direct physical impact. Examples: a closed sign, an announcement, an empty shelf.\n\n"
    + _RENDER_CORE
)

_REPAIR_SYSTEM = (
    "Fix the sentence so that:\n"
    "- The shared event is NEUTRAL (an external happening, third-party action,\n"
    "  announcement, arrival, rule) — not Agent A deliberately acting against\n"
    "  Agent B. If the current sentence has A punishing, coercing, taking from,\n"
    "  or reporting B to get something, rewrite with a neutral cause (e.g.,\n"
    "  a clerk's decision, a posted result, a shelf running out). Pride must\n"
    "  come from constructive achievement, never from dominating B.\n"
    "- A SINGLE event causes BOTH emotions. Agent A's positive feeling and\n"
    "  Agent B's negative feeling must both be consequences of the SAME\n"
    "  event (named in cause_span). If the current sentence has two\n"
    "  unrelated triggers (e.g., A excited about X, B disgusted by unrelated Y),\n"
    "  rewrite so a single event causes both outcomes.\n"
    "- Exactly one positive and one negative emotion (from the taxonomy).\n"
    "- Agent A = positive, Agent B = negative.\n"
    "- No emotion words or behavioral cues in sentence.\n"
    "- Agent A and Agent B have DIFFERENT role descriptions that uniquely identify each person.\n"
    "  E.g., instead of 'a student taking the test' vs 'another student taking the test',\n"
    "  use 'the student who passed the CPR test' vs 'the student who failed the CPR test'.\n"
    "- Role descriptions must use the SAME identifier as the sentence. If the sentence\n"
    "  names a character (e.g., 'Maya', 'Alex'), the role must reference that name. Do\n"
    "  NOT introduce a relationship label ('Fiancé', 'Roommate') that is absent from\n"
    "  the sentence — the annotator cannot link it to a named character.\n"
    "- NEVER use literal 'Agent A' or 'Agent B' in the sentence — use character names or descriptions.\n"
    "- The sentence must NOT contain any words that directly reveal the emotion.\n"
    "  No synonyms of the target emotion (e.g., no 'thank' if gratitude, no 'proud' if pride).\n"
    "  Show the situation; let the reader infer the feeling.\n"
    "- Keep the sentence under 300 characters if possible. Be concise.\n"
    "- Keep at most 3 distinct people in the scene. If the source has more,\n"
    "  collapse minor characters into 'the parents', 'her friends', etc., or\n"
    "  drop them — only the two contrast agents and one optional third party\n"
    "  (a clerk, a child, an audience reference) should be named or referred to.\n"
    "- AGENT ROLE HEADS MUST DIVERGE: the first 4 words / first clause of\n"
    "  agent_A.role and agent_B.role must be DISTINCT. Annotators may see\n"
    "  only the head noun phrase ('Maya', 'the trainee') and need to tell A\n"
    "  from B at a glance. Do NOT have both roles start with 'the student' /\n"
    "  'the customer' / 'the driver'. Either use proper names ('Maya' / 'Al')\n"
    "  or include a distinguishing modifier in the first few words\n"
    "  ('the senior nurse' / 'the new nurse', 'the cashier' / 'the manager').\n"
    "- Each agent's reason for their emotion is explicit in the sentence.\n"
    "- DISAMBIGUATION: relief requires a prior threat shown; gratitude requires a helper;\n"
    "  pride requires personal effort; excitement requires future event; anger requires a wrongdoer;\n"
    "  sadness requires no one to blame; fear requires unresolved threat; disgust requires morally revolting behavior;\n"
    "  embarrassment requires public audience watching.\n"
)

# ── Model call ────────────────────────────────────────────────────────────────


def schema_call(messages, schema, max_tokens=5000, retries=3, temperature=None):
    temp = temperature if temperature is not None else TEMPERATURE
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": schema},
                temperature=temp,
                max_completion_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(2)
    raise RuntimeError("Model failed repeatedly.")


# ── Validation ────────────────────────────────────────────────────────────────


ROLE_STOPWORDS = {
    "a", "an", "the", "of", "to", "and", "in", "who", "is", "for",
    "her", "his", "their", "with", "at", "on", "that", "same", "also",
    "has", "been", "are", "was", "be", "by", "from",
}

EMOTION_LEAKAGE = {
    "joy": ["happy", "happily", "glad", "pleased", "delighted", "cheerful", "joyful"],
    "pride": ["proud", "proudly", "accomplished", "achievement", "pride"],
    "relief": ["relieved", "relief", "sigh of relief", "weight off"],
    "gratitude": ["thank", "thanks", "thanked", "thankful", "grateful", "gratitude", "appreciate", "appreciated"],
    "excitement": ["excited", "excitement", "thrilled", "eager", "eagerly", "can't wait"],
    "anger": ["angry", "anger", "furious", "mad", "enraged", "rage", "outraged"],
    "sadness": ["sad", "sadly", "sadness", "depressed", "heartbroken", "devastated", "sorrow"],
    "fear": ["afraid", "scared", "terrified", "dread", "panic", "fearful", "fear"],
    "disgust": ["disgusted", "disgust", "disgusting", "revolted", "repulsed", "sickened", "revolting"],
    "embarrassment": ["embarrassed", "embarrassment", "humiliated", "ashamed", "mortified", "shame"],
}


def validate_version(blob, label):
    problems = []
    A = blob["agent_A"]["emotion"]
    B = blob["agent_B"]["emotion"]
    sent = blob["sentence"]
    sent_lower = sent.lower()

    # Emotion contrast check
    if not has_contrast(A, B):
        problems.append(f"{label}: not contrasting: A={A}, B={B}")
    if A not in POSITIVE:
        problems.append(f"{label}: Agent A must be positive, got {A}")
    if B not in NEGATIVE:
        problems.append(f"{label}: Agent B must be negative, got {B}")

    # Fix 1: No literal "Agent A" / "Agent B" in sentence
    if "agent a" in sent_lower or "agent b" in sent_lower:
        problems.append(
            f"{label}: sentence contains literal 'Agent A' or 'Agent B' — "
            "must use character names or descriptions instead"
        )

    # Fix 2: Roles must be distinguishable
    ra_words = set(blob["agent_A"]["role"].lower().split()) - ROLE_STOPWORDS
    rb_words = set(blob["agent_B"]["role"].lower().split()) - ROLE_STOPWORDS
    overlap = ra_words & rb_words
    if blob["agent_A"]["role"].lower() == blob["agent_B"]["role"].lower():
        problems.append(f"{label}: Agent A and B have identical roles")
    elif len(overlap) > 2:
        problems.append(
            f"{label}: roles too similar (overlap: {overlap}) — "
            "each role must uniquely identify the person by their outcome or trait"
        )

    # Fix 3: Emotion leakage — check both agents' emotions
    if contains_banned(sent):
        problems.append(f"{label}: banned words in sentence")
    for agent_key, emo in [("agent_A", A), ("agent_B", B)]:
        leakage_words = EMOTION_LEAKAGE.get(emo, [])
        for lw in leakage_words:
            if lw.lower() in sent_lower:
                problems.append(
                    f"{label}: emotion leakage — sentence contains '{lw}' "
                    f"which reveals {agent_key}'s emotion ({emo})"
                )
                break  # one leakage per agent is enough

    # Fix 4: Sentence length
    if len(sent) > 300:
        problems.append(
            f"{label}: sentence too long ({len(sent)} chars) — "
            "simplify to under 300 characters while keeping both agents' outcomes clear"
        )

    # Fix 5: SINGLE CAUSE — both emotions must stem from one shared event
    cause_rel = (blob.get("cause_effect_relation") or "")
    cause_rel_lower = cause_rel.lower()
    cause_span = (blob.get("cause_span") or "").strip()
    evidence_a = (blob.get("evidence_A") or "")
    evidence_b = (blob.get("evidence_B") or "")

    # Heuristic 1: two-cause phrasing in cause_effect_relation.
    # Common giveaway: "while Agent B's ..." or "while B's ..." or two "because" clauses.
    two_cause_patterns = [
        r"while\s+(?:agent\s+)?b['\u2019]s?\b",
        r"while\s+b['\u2019]s?\b",
        r",\s*while\s+[^,]{3,40}\s+(?:comes from|is driven by|is caused by|is from|stems from)",
    ]
    for pat in two_cause_patterns:
        if re.search(pat, cause_rel_lower):
            problems.append(
                f"{label}: cause_effect_relation describes TWO separate causes "
                f"(matched pattern '{pat}'). A single event must cause both emotions."
            )
            break

    # Heuristic 2: two "because" clauses strongly suggests two independent causes.
    if cause_rel_lower.count(" because ") >= 2:
        problems.append(
            f"{label}: cause_effect_relation contains multiple 'because' clauses — "
            "rewrite so one shared event causes both emotions."
        )

    # Heuristic 3 (soft): cause_span must appear in the sentence (so it's a real
    # text span). If cause_span is empty or absent from the sentence, that's a red flag.
    if cause_span:
        if cause_span.lower() not in sent.lower():
            problems.append(
                f"{label}: cause_span '{cause_span[:60]}...' is not a substring of the "
                "sentence — cause_span must quote a shared-event phrase from the sentence."
            )
    else:
        problems.append(f"{label}: cause_span is empty — needs a phrase naming the shared event.")

    # Fix 6: Crowded scene — too many distinct people in the sentence.
    # noun_count_validator lives in the same directory in this folder.
    try:
        from noun_count_validator import analyze as _noun_analyze
        _nc = _noun_analyze(sent)
        if _nc["person_count"] > 3:
            problems.append(
                f"{label}: too many person references "
                f"({_nc['person_count']}, refs={_nc['person_refs']}) — "
                "simplify the scene so it has at most 3 distinct people."
            )
    except Exception:
        pass

    # Fix 7: Agent-name collision under name-only display.
    # If the two role descriptions reduce to the same head noun under the
    # annotation-time extractor, the annotator can't tell them apart in
    # the agent-name-only view. Force the roles to differ in their first
    # 4 words.
    role_a_str = (blob["agent_A"]["role"] or "")
    role_b_str = (blob["agent_B"]["role"] or "")
    def _norm_head(s):
        # Take everything before first comma / "who"/"whose"/"that"; strip punct + lowercase
        s = (s or "").strip()
        cut_comma = s.find(",")
        m = re.search(r"\b(?:who|whose|that)\b", s, re.IGNORECASE)
        cuts = [c for c in [cut_comma if cut_comma > 0 else None,
                            m.start() if m else None] if c]
        head = s[: min(cuts)] if cuts else " ".join(s.split()[:4])
        return re.sub(r"[^\w\s]", "", head).strip().lower()
    head_a = _norm_head(role_a_str)
    head_b = _norm_head(role_b_str)
    if head_a and head_b and (head_a == head_b
                              or head_a.startswith(head_b + " ")
                              or head_b.startswith(head_a + " ")):
        problems.append(
            f"{label}: agent role HEADS would collide under name-only "
            f"display ('{head_a}' vs '{head_b}') — the first 4 words / "
            "first clause of each role must be distinct so the annotator "
            "can tell A and B apart at a glance."
        )

    return problems


def render_version(draft, *, render_prompt, schema, label, emotion_hint=""):
    prompt = render_prompt
    if emotion_hint:
        prompt += f"\n\nEMOTION GUIDANCE: {emotion_hint}"

    MAX_ATTEMPTS = 3
    best_blob = None
    best_issues = None

    for attempt in range(MAX_ATTEMPTS):
        blob = schema_call(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(draft, ensure_ascii=False)},
            ],
            schema=schema,
            max_tokens=5000,
        )

        issues = validate_version(blob, label)
        if not issues:
            return blob

        # Keep track of best attempt (fewest issues)
        if best_blob is None or len(issues) < len(best_issues):
            best_blob = blob
            best_issues = issues

        print(f"  [{label}] Attempt {attempt+1}/{MAX_ATTEMPTS}: {issues}")

        # Try repair
        blob = schema_call(
            messages=[
                {"role": "system", "content": _REPAIR_SYSTEM},
                {"role": "user", "content": "Problems:\n" + "\n".join(issues)},
                {"role": "user", "content": "Current JSON:\n" + json.dumps(blob, ensure_ascii=False)},
            ],
            schema=schema,
            max_tokens=5000,
        )

        issues = validate_version(blob, label)
        if not issues:
            return blob

        if len(issues) < len(best_issues):
            best_blob = blob
            best_issues = issues

    print(f"  [{label}] WARNING: could not fix all issues after {MAX_ATTEMPTS} attempts. Remaining: {best_issues}")
    return best_blob


def _build_category_block(cat):
    lines = [f"Pattern: {cat['description']}", "", "Examples:"]
    for ex in cat["examples"]:
        lines.append(f"  - {ex}")
    return "\n".join(lines)


# ── Distribution balancing ────────────────────────────────────────────────────


def get_emotion_guidance(pos_dist: Counter, neg_dist: Counter, target: float) -> str:
    """Suggest under-represented emotions."""
    pos_under = [e for e in POSITIVE if pos_dist[e] < target * 0.7]
    neg_under = [e for e in NEGATIVE if neg_dist[e] < target * 0.7]

    hints = []
    if pos_under:
        hints.append(f"Try to use one of these POSITIVE emotions (under-represented): {', '.join(pos_under)}")
    if neg_under:
        hints.append(f"Try to use one of these NEGATIVE emotions (under-represented): {', '.join(neg_under)}")

    return " | ".join(hints) if hints else ""


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    with open(REDDIT_DUMP, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Target: ~5 uses per emotion (25 stories × 2 versions = 50 assignments per polarity)
    target_per_emo = (NUM_STORIES * 2) / 5.0  # = 10
    pos_dist = Counter()
    neg_dist = Counter()

    # Detect already-generated stories to resume
    from glob import glob as _glob
    existing = {os.path.basename(p).split("_")[1] for p in _glob(os.path.join(OUTPUT_DIR, "story_*.json"))}

    # Auto-pick a story_idx base that doesn't collide with existing files.
    # Override via STORY_IDX_BASE env var if needed.
    if "STORY_IDX_BASE" in os.environ:
        idx_base = int(os.environ["STORY_IDX_BASE"])
    elif existing:
        idx_base = max(int(x) for x in existing if x.isdigit()) + 1
    else:
        idx_base = 0

    for i in range(NUM_STORIES):
        story_idx = i + idx_base
        line_idx = REDDIT_OFFSET + i
        if line_idx >= len(lines):
            print(f"[WARN] Only {len(lines)} stories in dump, stopping at {i}")
            break
        if str(story_idx) in existing:
            continue  # already done

        post = json.loads(lines[line_idx])
        story_text = post.get("selftext", "")
        title = post.get("title", f"story_{i}")

        category_phys = random.choice(CONTRASTIVE_CATEGORIES)
        category_nonphys = random.choice(CONTRASTIVE_CATEGORIES)

        emotion_guidance = get_emotion_guidance(pos_dist, neg_dist, target_per_emo)

        print(f"\n[{i+1}/{NUM_STORIES}] {title[:60]}...")
        print(f"  Categories: {category_phys['name']} / {category_nonphys['name']}")
        if emotion_guidance:
            print(f"  Guidance: {emotion_guidance}")

        # Draft for physical
        draft_prompt = DRAFT_PROMPT_TEMPLATE.format(
            category_block=_build_category_block(category_phys),
            emotion_guidance=emotion_guidance,
        )
        draft_phys = schema_call(
            messages=[
                {"role": "system", "content": draft_prompt},
                {"role": "user", "content": f"Story inspiration:\n{story_text[:2000]}"},
            ],
            schema=DRAFT_SCHEMA,
            max_tokens=5000,
            temperature=1.0,
        )

        # Render physical
        phys = render_version(
            draft_phys,
            render_prompt=PHYS_RENDER_PROMPT,
            schema=PHYS_SCHEMA,
            label="physical",
            emotion_hint=emotion_guidance,
        )
        pos_dist[phys["agent_A"]["emotion"]] += 1
        neg_dist[phys["agent_B"]["emotion"]] += 1

        # Draft for non-physical
        draft_prompt_np = DRAFT_PROMPT_TEMPLATE.format(
            category_block=_build_category_block(category_nonphys),
            emotion_guidance=get_emotion_guidance(pos_dist, neg_dist, target_per_emo),
        )
        draft_nonphys = schema_call(
            messages=[
                {"role": "system", "content": draft_prompt_np},
                {"role": "user", "content": f"Story inspiration:\n{story_text[:2000]}"},
            ],
            schema=DRAFT_SCHEMA,
            max_tokens=5000,
            temperature=1.0,
        )

        # Render non-physical
        nonphys = render_version(
            draft_nonphys,
            render_prompt=NONPHYS_RENDER_PROMPT,
            schema=NONPHYS_SCHEMA,
            label="non_physical",
            emotion_hint=get_emotion_guidance(pos_dist, neg_dist, target_per_emo),
        )
        pos_dist[nonphys["agent_A"]["emotion"]] += 1
        neg_dist[nonphys["agent_B"]["emotion"]] += 1

        # Save
        final = {
            "category_physical": category_phys["name"],
            "category_non_physical": category_nonphys["name"],
            "setting_physical": draft_phys["setting"],
            "setting_non_physical": draft_nonphys["setting"],
            "physical_version": phys,
            "non_physical_version": nonphys,
        }

        safe_setting = re.sub(r'[\\/:*?"<>|]', "-", draft_phys["setting"][:80])
        outpath = os.path.join(OUTPUT_DIR, f"story_{story_idx}_{safe_setting}.json")
        with open(outpath, "w", encoding="utf-8") as out:
            json.dump(final, out, indent=2, ensure_ascii=False)

        print(f"  Saved: {os.path.basename(outpath)}")
        print(f"  Pos: {dict(pos_dist)} | Neg: {dict(neg_dist)}")

    print(f"\n[OK] Generated {NUM_STORIES} stories -> {OUTPUT_DIR}/")
    print(f"  Positive distribution: {dict(pos_dist)}")
    print(f"  Negative distribution: {dict(neg_dist)}")


if __name__ == "__main__":
    random.seed(42)
    main()
