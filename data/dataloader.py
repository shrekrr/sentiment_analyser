# dataloader.py
# Combines all four datasets into a single multi-task DataLoader.
# Uses proportional sampling so no single dataset dominates training.

import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from data.datasets import SST2Dataset, TweetSentimentDataset, GoEmotionsDataset, ISarcasmDataset


def build_dataloaders(batch_size: int = 16) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).

    Sampling strategy:
    - SST-2 train: 67k samples  → weight = 1.0 (baseline)
    - TweetSentiment train: 45k → weight = 1.0
    - GoEmotions train: 43k     → weight = 1.2 (slight upweight — emotion head needs signal)
    - iSarcasm train: ~3.5k     → weight = 8.0 (heavily upweight — tiny dataset)

    We sample uniformly with replacement so every epoch sees a roughly balanced
    mix regardless of raw dataset sizes. This is critical for the sarcasm head —
    without upweighting, the model sees iSarcasm samples too rarely to learn.
    """

    print("Loading datasets...")

    # ── Train splits ──────────────────────────────────────────────────────────
    sst2_train   = SST2Dataset(split="train")
    tweet_train  = TweetSentimentDataset(split="train")
    goemo_train  = GoEmotionsDataset(split="train")
    sarc_train   = ISarcasmDataset(split="train")

    # ── Val splits ────────────────────────────────────────────────────────────
    sst2_val     = SST2Dataset(split="validation")
    tweet_val    = TweetSentimentDataset(split="validation")
    goemo_val    = GoEmotionsDataset(split="validation")
    sarc_val     = ISarcasmDataset(split="test")  # iSarcasm has no "validation" split

    print(f"  SST-2        train={len(sst2_train):,}  val={len(sst2_val):,}")
    print(f"  TweetSent    train={len(tweet_train):,}  val={len(tweet_val):,}")
    print(f"  GoEmotions   train={len(goemo_train):,}  val={len(goemo_val):,}")
    print(f"  iSarcasmEval train={len(sarc_train):,}  val={len(sarc_val):,}")

    # ── Combine + build sample weights ───────────────────────────────────────
    dataset_configs = [
        (sst2_train,  1.0),
        (tweet_train, 1.0),
        (goemo_train, 1.2),
        (sarc_train,  8.0),
    ]

    train_combined = ConcatDataset([d for d, _ in dataset_configs])
    val_combined   = ConcatDataset([sst2_val, tweet_val, goemo_val, sarc_val])

    # Build per-sample weights for WeightedRandomSampler
    weights = []
    for dataset, w in dataset_configs:
        weights.extend([w] * len(dataset))
    weights = torch.tensor(weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_combined),
        replacement=True,
    )

    train_loader = DataLoader(
        train_combined,
        batch_size=batch_size,
        sampler=sampler,       # WeightedRandomSampler replaces shuffle=True
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_combined,
        batch_size=batch_size * 2,  # larger batch fine for eval (no grad)
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"\nTrain batches per epoch: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")

    return train_loader, val_loader


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader = build_dataloaders(batch_size=16)

    print("\nSample batch keys and shapes:")
    batch = next(iter(train_loader))
    for k, v in batch.items():
        print(f"  {k}: {v.shape}  dtype={v.dtype}")

    # Verify ignore indices are present (some tasks will have IGNORE=-1)
    print(f"\n  Unique sentiment labels in batch: {batch['sentiment'].unique().tolist()}")
    print(f"  Unique sarcasm labels in batch:   {batch['sarcasm'].unique().tolist()}")
    print(f"  Emotion vector sample:            {batch['emotions'][0].tolist()}")

    print("\nData pipeline looks good.")
