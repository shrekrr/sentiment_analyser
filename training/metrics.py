import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss

def get_ece(y_true, y_pred_class, y_pred_conf, n_bins=15):
    """
    Computes the Expected Calibration Error (ECE).
    """
    bins = np.linspace(0., 1., n_bins + 1)
    bin_indices = np.digitize(y_pred_conf, bins, right=True)
    
    ece = 0.0
    N = len(y_true)
    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        bin_n = np.sum(mask)
        if bin_n > 0:
            bin_acc = np.mean(y_true[mask] == y_pred_class[mask])
            bin_conf = np.mean(y_pred_conf[mask])
            ece += np.abs(bin_acc - bin_conf) * (bin_n / N)
    return float(ece)

def compute_metrics(all_outputs: list[dict], all_batches: list[dict]) -> dict:
    """
    Computes all validation metrics across tasks.
    """
    # Sentiment Lists
    sent_preds, sent_confs, sent_trues, sarc_trues_for_sent = [], [], [], []
    
    # Emotion Lists
    emo_preds, emo_trues = [], []
    
    # Sarcasm Lists
    sarc_preds, sarc_trues = [], []
    
    for out, batch in zip(all_outputs, all_batches):
        # -- Sentiment --
        s_logits = out["sentiment_logits"]
        s_probs = torch.softmax(s_logits, dim=-1)
        s_conf, s_pred = torch.max(s_probs, dim=-1)
        
        s_mask = batch["sentiment"] != -1
        if s_mask.any():
            sent_preds.extend(s_pred[s_mask].cpu().numpy())
            sent_confs.extend(s_conf[s_mask].cpu().numpy())
            sent_trues.extend(batch["sentiment"][s_mask].cpu().numpy())
            sarc_trues_for_sent.extend(batch["sarcasm"][s_mask].cpu().numpy())
            
        # -- Emotion --
        e_logits = out["emotion_logits"]
        e_probs = torch.sigmoid(e_logits)
        e_pred = (e_probs > 0.5).long()
        e_mask = (batch["emotions"] != -1).any(dim=1)
        if e_mask.any():
            emo_preds.extend(e_pred[e_mask].cpu().numpy())
            emo_trues.extend(batch["emotions"][e_mask].cpu().numpy())
            
        # -- Sarcasm --
        sarc_logits = out["sarcasm_logits"]
        sarc_pred = torch.argmax(sarc_logits, dim=-1)
        sarcasm_mask = batch["sarcasm"] != -1
        if sarcasm_mask.any():
            sarc_preds.extend(sarc_pred[sarcasm_mask].cpu().numpy())
            sarc_trues.extend(batch["sarcasm"][sarcasm_mask].cpu().numpy())
            
    # Calculate scores
    metrics = {}
    
    # Sentiment Metrics
    if len(sent_trues) > 0:
        sent_t = np.array(sent_trues)
        sent_p = np.array(sent_preds)
        sent_c = np.array(sent_confs)
        sarc_mask = np.array(sarc_trues_for_sent) == 1
        nonsarc_mask = np.array(sarc_trues_for_sent) == 0
        
        metrics["sentiment_f1_macro"] = float(f1_score(sent_t, sent_p, average="macro"))
        metrics["sentiment_accuracy"] = float(accuracy_score(sent_t, sent_p))
        metrics["ece"] = get_ece(sent_t, sent_p, sent_c)
        
        if np.any(sarc_mask):
            metrics["sentiment_accuracy_sarcastic"] = float(accuracy_score(sent_t[sarc_mask], sent_p[sarc_mask]))
        else:
            metrics["sentiment_accuracy_sarcastic"] = 0.0
             
        if np.any(nonsarc_mask):
            metrics["sentiment_accuracy_nonsarcastic"] = float(accuracy_score(sent_t[nonsarc_mask], sent_p[nonsarc_mask]))
        else:
            metrics["sentiment_accuracy_nonsarcastic"] = 0.0
    else:
        metrics.update({"sentiment_f1_macro": 0.0, "sentiment_accuracy": 0.0, "ece": 0.0, "sentiment_accuracy_sarcastic": 0.0, "sentiment_accuracy_nonsarcastic": 0.0})

    # Emotion Metrics
    if len(emo_trues) > 0:
        emo_t = np.array(emo_trues)
        emo_p = np.array(emo_preds)
        metrics["emotion_f1_macro"] = float(f1_score(emo_t, emo_p, average="macro"))
        metrics["emotion_hamming_loss"] = float(hamming_loss(emo_t, emo_p))
    else:
        metrics.update({"emotion_f1_macro": 0.0, "emotion_hamming_loss": 0.0})
        
    # Sarcasm Metrics
    if len(sarc_trues) > 0:
        sarc_t = np.array(sarc_trues)
        sarc_p = np.array(sarc_preds)
        metrics["sarcasm_f1_macro"] = float(f1_score(sarc_t, sarc_p, average="macro"))
        metrics["sarcasm_precision"] = float(precision_score(sarc_t, sarc_p, average="binary", zero_division=0))
        metrics["sarcasm_recall"] = float(recall_score(sarc_t, sarc_p, average="binary", zero_division=0))
    else:
        metrics.update({"sarcasm_f1_macro": 0.0, "sarcasm_precision": 0.0, "sarcasm_recall": 0.0})

    metrics["overall_score"] = (metrics.get("sentiment_f1_macro", 0.0) + 
                                metrics.get("emotion_f1_macro", 0.0) + 
                                metrics.get("sarcasm_f1_macro", 0.0)) / 3.0

    return metrics
