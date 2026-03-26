import torch
import torch.nn as nn
from transformers import DistilBertModel
from .heads import SentimentHead, EmotionHead, SarcasmHead

class MultiTaskSentimentModel(nn.Module):
    def __init__(self, freeze_encoder_epochs=2):
        super().__init__()
        # Load pre-trained DistilBERT with eager attention to support output_attentions=True for explainability
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased", attn_implementation="eager")
        
        # Attach heads
        self.sentiment_head = SentimentHead()
        self.emotion_head = EmotionHead()
        self.sarcasm_head = SarcasmHead()
        
        self.freeze_encoder_epochs = freeze_encoder_epochs
        
    def forward(self, input_ids, attention_mask):
        # Run DistilBERT
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract [CLS] token representation (index 0 of last hidden state)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through all three heads
        sentiment_logits = self.sentiment_head(cls_output)
        emotion_logits = self.emotion_head(cls_output)
        sarcasm_logits = self.sarcasm_head(cls_output)
        
        return {
            "sentiment_logits": sentiment_logits,
            "emotion_logits": emotion_logits,
            "sarcasm_logits": sarcasm_logits
        }
        
    def freeze_encoder(self):
        """Called during Phase 1 of training to freeze the DistilBERT encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Called when Phase 2 begins to unfreeze the DistilBERT encoder for full fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
