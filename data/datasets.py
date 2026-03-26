# datasets.py
# One Dataset class per data source. Each returns a dict with whichever
# label tensors that source provides. Missing labels use -1 as ignore index.

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
from datasets import load_dataset
from data.label_maps import (
    SENTIMENT2ID, EMOTION2ID, goemotion_ids_to_plutchik_vector,
    SENTIMENT_LABELS, EMOTION_LABELS
)

TOKENIZER = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
MAX_LEN = 128
IGNORE = -1  # used as ignore_index in loss functions for missing labels


def tokenize(text: str) -> dict:
    return TOKENIZER(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


# ── 1. SST-2 (Sentiment) ───────────────────────────────────────────────────────
# Labels: 0 = negative, 1 = positive (binary)
# We map to our 3-class space: 0 = negative, 2 = positive (skip neutral)
SST2_MAP = {0: 0, 1: 2}  # negative->0, positive->2

class SST2Dataset(Dataset):
    """Stanford Sentiment Treebank v2. Provides: sentiment labels only."""

    def __init__(self, split="train"):
        # split: "train" | "validation"
        raw = load_dataset("glue", "sst2", split=split)
        # Filter out -1 label rows (test set has no labels in GLUE)
        self.data = [r for r in raw if r["label"] != -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        enc = tokenize(row["sentence"])
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sentiment":      torch.tensor(SST2_MAP[row["label"]], dtype=torch.long),
            "emotions":       torch.tensor([IGNORE] * len(EMOTION_LABELS), dtype=torch.float),
            "sarcasm":        torch.tensor(IGNORE, dtype=torch.long),
        }


# ── 2. SemEval 2017 Task 4A (Sentiment – adds Neutral class) ──────────────────
# HuggingFace slug: "sem_eval_2018_task_1" subset "subtask5.english" won't work.
# Use tweet_eval which has the 3-class twitter sentiment.
# Labels: 0=negative, 1=neutral, 2=positive — matches our space directly.

class TweetSentimentDataset(Dataset):
    """Tweet-eval 3-class sentiment. Provides: sentiment labels only."""

    def __init__(self, split="train"):
        raw = load_dataset("tweet_eval", "sentiment", split=split)
        self.data = list(raw)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        enc = tokenize(row["text"])
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sentiment":      torch.tensor(row["label"], dtype=torch.long),
            "emotions":       torch.tensor([IGNORE] * len(EMOTION_LABELS), dtype=torch.float),
            "sarcasm":        torch.tensor(IGNORE, dtype=torch.long),
        }


# ── 3. GoEmotions (Emotion – 27 labels → Plutchik 8) ─────────────────────────
# Multi-label: each sample can have multiple emotions.
# We convert to a multi-hot Plutchik vector.

class GoEmotionsDataset(Dataset):
    """Google GoEmotions dataset. Provides: emotion labels only."""

    def __init__(self, split="train"):
        # "simplified" config already maps to 28 labels
        raw = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
        self.data = list(raw)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        # row["labels"] is a list of active label indices
        emotion_vec = goemotion_ids_to_plutchik_vector(row["labels"])
        enc = tokenize(row["text"])
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sentiment":      torch.tensor(IGNORE, dtype=torch.long),
            "emotions":       torch.tensor(emotion_vec, dtype=torch.float),
            "sarcasm":        torch.tensor(IGNORE, dtype=torch.long),
        }


# ── 4. iSarcasmEval 2022 (Sarcasm) ───────────────────────────────────────────
# Binary: 0 = not sarcastic, 1 = sarcastic
# HuggingFace slug: "Iarruf/iSarcasmEval" — small dataset (~4k samples)
# Also carries a sentiment label we can optionally use.

class ISarcasmDataset(Dataset):
    """iSarcasmEval 2022. Provides: sarcasm labels. Sentiment if available."""

    def __init__(self, split="train"):
        raw = load_dataset("viethq1906/isarcasm_2022_taskA_En", split=split)
        self.data = list(raw)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        enc = tokenize(row["sentence"])

        # We only use the binary sarcasm label for reliability (mapped to "sentiment" string in this dataset)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sentiment":      torch.tensor(IGNORE, dtype=torch.long),
            "emotions":       torch.tensor([IGNORE] * len(EMOTION_LABELS), dtype=torch.float),
            "sarcasm":        torch.tensor(int(row["sentiment"]), dtype=torch.long),
        }
