import torch
import numpy as np

def get_attention_heatmap(model, tokenizer, text: str) -> dict:
    """
    Extract last-layer attention weights from DistilBERT.
    Average across all 12 attention heads.
    Return:
      {
        "tokens": list[str],
        "attention_weights": list[float],
        "raw_attention_matrix": np.ndarray
      }
    """
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        # output_attentions=True requests the self-attention weights from the transformer
        outputs = model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True
        )
        # outputs.attentions is a tuple of 6 tensors for DistilBERT (layer 0 to 5)
        # each tensor shape: (batch_size, num_heads, seq_len, seq_len)
        last_layer_att = outputs.attentions[-1]
        
    # We examine the attention *received* by each token from the [CLS] token (index 0).
    # last_layer_att[batch_idx, head_idx, from_idx, to_idx]
    cls_attention = last_layer_att[0, :, 0, :]  # Shape: [num_heads, seq_len]
    
    # Average across all heads
    avg_attention = cls_attention.mean(dim=0).cpu().numpy()  # Shape: [seq_len]
    
    # Normalize weights to range 0.0 - 1.0
    val_min, val_max = avg_attention.min(), avg_attention.max()
    if val_max > val_min:
        norm_attention = (avg_attention - val_min) / (val_max - val_min)
    else:
        norm_attention = avg_attention

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    return {
        "tokens": tokens,
        "attention_weights": norm_attention.tolist(),
        "raw_attention_matrix": last_layer_att[0].cpu().numpy()
    }
