# Multi-Task Sentiment Analysis System

A comprehensive multi-task sentiment analysis engine fine-tuned from `distilbert-base-uncased`. Capable of predicting overall sentiment, Plutchik 8 emotions (multilabel), and structural sarcasm using a frozen-encoder-first discriminative fine-tuning strategy. Includes fully configured Explainable AI pipelines (SHAP, layer-wise attention) and a premium FastAPI/VanillaJS dashboard.

## Architecture

```text
Input Text  ────────> [ tokenizer ] ────────> [ DistilBERT Encoder ]
                                                        │
                                                        ▼
                                                  [CLS] Token Representation
                                                        │
                 ┌──────────────────────────────────────┼────────────────────────────┐
                 ▼                                      ▼                            ▼
        [ Sentiment Head ]                      [ Emotion Head ]                [ Sarcasm Head ]
        - Dropout(0.3)                          - Dropout(0.3)                  - Dropout(0.3)
        - Linear(256) + GELU                    - Linear(256) + GELU            - Linear(128) + GELU
        - Linear(3)                             - Linear(8)                     - Linear(2)
                 │                                      │                            │
                 ▼                                      ▼                            ▼
         Negative/Neutral/Positive               Plutchik 8 Emotions              Binary Sarcasm
```

## Setup Instructions

### Environment
```bash
pip install -r requirements.txt
```

### 1. Training & Ablation Experiments
The project uses JSON config files within `configs/` to define training runs. To launch the full model training loop:
```bash
python -m training.train --config configs/full_model.json
```

Other available ablation studies constraints:
- No Sarcasm penalty:
  ```bash
  python -m training.train --config configs/no_sarcasm.json
  ```
- No Emotion labels:
  ```bash
  python -m training.train --config configs/no_emotion.json
  ```
- Frozen Encoder baseline:
  ```bash
  python -m training.train --config configs/frozen_encoder.json
  ```

### 2. Live Frontend Demo
To spin up the web application dashboard API, configure the checkpoint variable to load the `best.pt` file created during output.
```bash
export MODEL_CHECKPOINT=checkpoints/full_model_best.pt
python -m frontend.app
# OR
uvicorn frontend.app:app --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) (the app falls back on `/` mount).

## Notebooks
Open Jupyter Notebook to visualize ablation metrics across the 4 experiment types:
```bash
jupyter notebook notebooks/ablation_results.ipynb
```

## Explainability
The inference endpoints utilize integrated `shap` computations and DistilBERT layer-12 output attentions for per-token relevance visualizations rendered dynamically in the UI dashboard. `shap_explain.py` wraps the fast tokenizer into computationally intensive masks mapped statically for caching.

## References
Datasets aggregated spanning: 
- `tweet_eval`
- `google-research-datasets/go_emotions` 
- `Iarruf/iSarcasmEval`
- Stanford Sentiment Treebank
