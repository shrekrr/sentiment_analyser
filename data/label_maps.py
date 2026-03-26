# label_maps.py
# Defines all label spaces and the GoEmotions -> Plutchik collapse mapping

# ── Sentiment ──────────────────────────────────────────────────────────────────
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT2ID = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
ID2SENTIMENT = {i: l for l, i in SENTIMENT2ID.items()}

# ── Plutchik 8 emotions ────────────────────────────────────────────────────────
EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
EMOTION2ID = {l: i for i, l in enumerate(EMOTION_LABELS)}
ID2EMOTION = {i: l for l, i in EMOTION2ID.items()}

# ── GoEmotions 28 labels (27 emotions + neutral) ──────────────────────────────
# Source: https://huggingface.co/datasets/google-research-datasets/go_emotions
GOEMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Maps each GoEmotion label -> Plutchik emotion (or None to drop)
# Methodological note: "neutral" and ambiguous labels are dropped from
# multi-label training but the sample is still kept for other task heads.
GOEMOTION_TO_PLUTCHIK = {
    # Joy: positive high-arousal states
    "admiration":    "joy",
    "amusement":     "joy",
    "excitement":    "joy",
    "gratitude":     "joy",
    "joy":           "joy",
    "love":          "joy",
    "optimism":      "joy",
    "pride":         "joy",
    "relief":        "joy",

    # Trust: approval and caring
    "approval":      "trust",
    "caring":        "trust",

    # Anticipation: curiosity and desire
    "curiosity":     "anticipation",
    "desire":        "anticipation",

    # Sadness: grief, loss, regret
    "disappointment": "sadness",
    "grief":          "sadness",
    "remorse":        "sadness",
    "sadness":        "sadness",

    # Anger
    "anger":         "anger",
    "annoyance":     "anger",
    "disapproval":   "anger",

    # Disgust
    "disgust":       "disgust",
    "embarrassment": "disgust",

    # Fear
    "fear":          "fear",
    "nervousness":   "fear",

    # Surprise
    "confusion":     "surprise",
    "realization":   "surprise",
    "surprise":      "surprise",

    # Drop: no clean Plutchik mapping
    "neutral":       None,
}

def goemotion_ids_to_plutchik_vector(goemotion_ids: list[int]) -> list[float]:
    """
    Convert a list of GoEmotions label indices to a multi-hot Plutchik vector.
    Returns a float list of length 8 (one per Plutchik emotion).
    Multiple GoEmotions can activate the same Plutchik slot — we clip to 1.
    """
    vector = [0.0] * len(EMOTION_LABELS)
    for idx in goemotion_ids:
        label = GOEMOTION_LABELS[idx]
        plutchik = GOEMOTION_TO_PLUTCHIK.get(label)
        if plutchik is not None:
            vector[EMOTION2ID[plutchik]] = 1.0
    return vector
