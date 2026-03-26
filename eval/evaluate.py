import argparse
import json
import torch
from model.model import MultiTaskSentimentModel
from data.dataloader import build_dataloaders
from training.metrics import compute_metrics
from tqdm import tqdm

def evaluate(checkpoint_path: str, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MultiTaskSentimentModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Get only validation loader
    _, val_loader = build_dataloaders(batch_size=batch_size)
    
    all_outputs = []
    all_batches = []
    
    print(f"Evaluating {checkpoint_path} on {device}...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=batch_dev["input_ids"], attention_mask=batch_dev["attention_mask"])
            
            # Store on CPU for metrics calculation
            batch_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
            
            all_batches.append(batch_cpu)
            all_outputs.append(out_cpu)
            
    metrics = compute_metrics(all_outputs, all_batches)
    
    print("\n--- Evaluation Results ---")
    print(json.dumps(metrics, indent=4))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best model .pt")
    args = parser.parse_args()
    evaluate(args.checkpoint)
