# Data Pipeline

## File structure

```
data/
├── label_maps.py      # All label spaces + GoEmotions → Plutchik collapse
├── datasets.py        # One Dataset class per data source
├── dataloader.py      # Combined multi-task DataLoader with weighted sampling
└── README.md
```

## Datasets used

| Task | Dataset | HuggingFace slug | Size |
|---|---|---|---|
| Sentiment | SST-2 | `glue/sst2` | ~67k train |
| Sentiment (neutral) | TweetEval | `tweet_eval/sentiment` | ~45k train |
| Emotion | GoEmotions | `google-research-datasets/go_emotions` | ~43k train |
| Sarcasm | iSarcasmEval | `Iarruf/iSarcasmEval` | ~3.5k train |

## Setup

```bash
pip install transformers datasets torch
```

## Run sanity check

```bash
cd data/
python dataloader.py
```

Expected output:
```
Loading datasets...
  SST-2        train=67,349  val=872
  TweetSent    train=45,615  val=2,000
  GoEmotions   train=43,410  val=5,426
  iSarcasmEval train=3,468   val=1,000

Train batches per epoch: 9,990
Val batches: 3,491

Sample batch keys and shapes:
  input_ids:      torch.Size([16, 128])  dtype=torch.int64
  attention_mask: torch.Size([16, 128])  dtype=torch.int64
  sentiment:      torch.Size([16])       dtype=torch.int64
  emotions:       torch.Size([16, 8])    dtype=torch.float32
  sarcasm:        torch.Size([16])       dtype=torch.int64

  Unique sentiment labels in batch: [-1, 0, 1, 2]
  Unique sarcasm labels in batch:   [-1, 0, 1]
```

## Key design decisions (document these in your report)

**1. GoEmotions label collapse**
GoEmotions has 27 emotion categories. We collapse them to Plutchik's 8 using
a manually constructed mapping in `label_maps.py`. The mapping is the first
methodological decision worth discussing — several GoEmotions labels are
ambiguous (e.g. "realization" could be surprise or anticipation). Our choice
to map it to surprise follows the intuition that realization involves
unexpectedness.

**2. Ignore index (-1)**
Each dataset only provides labels for its own task. Missing labels use -1 as
the ignore_index, which PyTorch's CrossEntropyLoss and BCEWithLogitsLoss
support natively. This means:
  - An SST-2 sample contributes to sentiment loss only
  - A GoEmotions sample contributes to emotion loss only
  - An iSarcasm sample contributes to sarcasm loss only

**3. Weighted random sampling**
iSarcasm is ~20x smaller than SST-2. Without upweighting, the sarcasm head
sees too few samples per epoch to learn effectively. We use
WeightedRandomSampler with a weight of 8.0 for iSarcasm samples, which
roughly equalizes the effective contribution of each dataset per epoch.
This is a hyperparameter worth ablating.

**4. Tokenizer**
We use DistilBertTokenizerFast with max_length=128. Tweets are short (<128
tokens almost always). SST-2 sentences are short. GoEmotions sentences are
short. The only risk is longer Reddit comments in the sarcasm corpus, which
we truncate. A max_length=256 experiment is worth noting as future work.
