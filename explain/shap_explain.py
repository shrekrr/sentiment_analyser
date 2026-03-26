import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# Internal cache so we don't recalculate SHAP on duplicate calls for the identical text
_SHAP_CACHE = {}

def explain_with_shap(model, tokenizer, texts: list[str], task: str = "sentiment", output_dir="frontend/static/shap_plots") -> list[dict]:
    """
    Use shap.Explainer with a pipeline wrapper.
    task: "sentiment" | "emotion" | "sarcasm"
    Returns SHAP values per token per class.
    Generates and saves waterfall plot for each text.
    """
    global _SHAP_CACHE
    device = next(model.parameters()).device
    model.eval()

    def predict_func(texts_in):
        if isinstance(texts_in, np.ndarray):
            texts_in = texts_in.tolist()
            
        inputs = tokenizer(texts_in, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            if task == "sentiment":
                logits = outputs["sentiment_logits"]
            elif task == "emotion":
                logits = outputs["emotion_logits"]
            elif task == "sarcasm":
                logits = outputs["sarcasm_logits"]
            else:
                raise ValueError(f"Unknown task {task}")
        return logits.cpu().numpy()

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_func, masker)
    
    results = []
    os.makedirs(output_dir, exist_ok=True)
    
    for text in texts:
        cache_key = f"{text}_{task}"
        
        if cache_key in _SHAP_CACHE:
            shap_values = _SHAP_CACHE[cache_key]
        else:
            # SHAP computation is slow
            shap_values = explainer([text])
            _SHAP_CACHE[cache_key] = shap_values
            
        # Evaluate to find predicted class for waterfall plot
        pred_logits = predict_func([text])[0]
        # For mult-label emotion we could pick max, or loop. We'll pick the most activated one.
        pred_class = int(np.argmax(pred_logits))
        
        plot_filename = f"shap_{task}_{abs(hash(text))}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        
        # shap_values is Explanation object. Shape: [1, num_tokens, num_classes]
        # Plot reasoning for the most strongly predicted class
        plt.figure()
        shap.waterfall_plot(shap_values[0, :, pred_class], show=False)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        results.append({
            "text": text,
            "tokens": list(shap_values[0].data),
            "values": shap_values[0].values[:, pred_class].tolist(),
            "predicted_class": pred_class,
            "plot_path": plot_filename
        })
        
    return results
