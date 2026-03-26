import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_sentiment=1.0, lambda_emotion=0.8, lambda_sarcasm=0.5):
        super().__init__()
        self.lambda_sentiment = lambda_sentiment
        self.lambda_emotion = lambda_emotion
        self.lambda_sarcasm = lambda_sarcasm
        
        # We use ignore_index=-1
        self.sentiment_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.sarcasm_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        # For emotion, we use BCEWithLogitsLoss, but we'll mask out -1 samples manually
        self.emotion_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, batch):
        """
        outputs: dict with "sentiment_logits", "emotion_logits", "sarcasm_logits"
        batch: dict with "sentiment", "emotions", "sarcasm"
        """
        # 1. Sentiment Loss
        if (batch["sentiment"] != -1).any():
            l_sent = self.sentiment_loss_fn(outputs["sentiment_logits"], batch["sentiment"])
        else:
            l_sent = torch.tensor(0.0, device=outputs["sentiment_logits"].device)
        
        # 2. Sarcasm Loss
        if (batch["sarcasm"] != -1).any():
            l_sarc = self.sarcasm_loss_fn(outputs["sarcasm_logits"], batch["sarcasm"])
        else:
            l_sarc = torch.tensor(0.0, device=outputs["sarcasm_logits"].device)
        
        # 3. Emotion Loss
        # Emotion shape: [batch, 8]
        # Ignore samples where ALL emotions are -1 (meaning no label provided)
        emotion_targets = batch["emotions"]
        emotion_mask = (emotion_targets != -1).any(dim=1)  # [batch] bool
        
        if emotion_mask.sum() > 0:
            valid_logits = outputs["emotion_logits"][emotion_mask]
            valid_targets = emotion_targets[emotion_mask]
            l_emo_all = self.emotion_loss_fn(valid_logits, valid_targets)
            l_emo = l_emo_all.mean()
        else:
            # no emotion supervised samples in batch
            l_emo = torch.tensor(0.0, device=outputs["sentiment_logits"].device, requires_grad=True)
            
        total_loss = (self.lambda_sentiment * l_sent + 
                      self.lambda_emotion * l_emo + 
                      self.lambda_sarcasm * l_sarc)
                      
        return {
            "total": total_loss,
            "sentiment": l_sent.detach() if l_sent.requires_grad else l_sent,
            "emotion": l_emo.detach() if l_emo.requires_grad else l_emo,
            "sarcasm": l_sarc.detach() if l_sarc.requires_grad else l_sarc,
        }
