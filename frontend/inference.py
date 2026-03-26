import torch
from transformers import DistilBertTokenizerFast
from model.model import MultiTaskSentimentModel
from explain.attention_viz import get_attention_heatmap
from data.label_maps import ID2SENTIMENT, ID2EMOTION

class SentimentInference:
    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        
        self.model = MultiTaskSentimentModel()
        import os
        if checkpoint_path and os.path.exists(checkpoint_path):
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            else:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        elif checkpoint_path:
            print(f"WARNING: Checkpoint missing at {checkpoint_path}! Using raw base model weights for inference.")
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(text, max_length=128, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            
            # 1. Sentiment
            s_logits = outputs["sentiment_logits"][0]
            s_probs = torch.softmax(s_logits, dim=-1)
            s_conf, s_pred = torch.max(s_probs, dim=-1)
            
            # 2. Emotion
            e_logits = outputs["emotion_logits"][0]
            e_probs = torch.sigmoid(e_logits)
            
            # 3. Sarcasm
            sarc_logits = outputs["sarcasm_logits"][0]
            sarc_probs = torch.softmax(sarc_logits, dim=-1)
            sarc_conf, sarc_pred = torch.max(sarc_probs, dim=-1)
            
        # Attention - run through heatmap exporter
        attention_data = get_attention_heatmap(self.model, self.tokenizer, text)
        contrastive_shift = self._detect_contrastive_shift(text)
        
        # Build sentiment dict
        sentiment_scores = {ID2SENTIMENT[i]: float(s_probs[i]) for i in range(3)}
        sentiment_label = ID2SENTIMENT[int(s_pred)]
        
        # Build emotion dict
        emotion_scores = {ID2EMOTION[i]: float(e_probs[i]) for i in range(8)}
        dominant_emotion_idx = int(torch.argmax(e_probs))
        dominant_emotion = ID2EMOTION[dominant_emotion_idx]
        
        # Sarcasm
        sarcasm_detected = bool(sarc_pred == 1)
        
        # Reasoning text
        reasoning = f"The model predicts a {sentiment_label} sentiment with {float(s_conf)*100:.1f}% confidence. "
        reasoning += f"The dominant emotion is {dominant_emotion}. "
        if sarcasm_detected:
            reasoning += "It also detected structural sarcasm, indicating the text's literal meaning may differ from its intended meaning. "
        if contrastive_shift:
            reasoning += "A contrastive shift was detected in the phrasing (e.g., 'but', 'however'), which typically indicates mixed polarity."

        return {
            "text": text,
            "sentiment": {
                "label": sentiment_label,
                "confidence": float(s_conf),
                "scores": sentiment_scores
            },
            "emotions": emotion_scores,
            "dominant_emotion": dominant_emotion,
            "sarcasm": {
                "detected": sarcasm_detected,
                "confidence": float(sarc_conf)
            },
            "attention": {
                "tokens": attention_data["tokens"],
                "weights": attention_data["attention_weights"]
            },
            "contrastive_shift": contrastive_shift,
            "reasoning": reasoning
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    def _detect_contrastive_shift(self, text: str) -> bool:
        CONTRAST_WORDS = ["but", "however", "although", "yet",
                          "nevertheless", "despite", "while", "whereas"]
        text_lower = text.lower()
        
        for word in CONTRAST_WORDS:
            if f" {word} " in f" {text_lower} ":
                return True
        return False
