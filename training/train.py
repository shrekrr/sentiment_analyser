import os
import json
import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from model.model import MultiTaskSentimentModel
from data.dataloader import build_dataloaders
from training.losses import MultiTaskLoss
from training.metrics import compute_metrics
from tqdm import tqdm
import pandas as pd

def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataloaders
    train_loader, val_loader = build_dataloaders(batch_size=config["batch_size"])
    
    # 2. Model & Loss
    model = MultiTaskSentimentModel(freeze_encoder_epochs=config["freeze_epochs"])
    model.to(device)
    
    criterion = MultiTaskLoss(
        lambda_sentiment=config["lambda_sentiment"],
        lambda_emotion=config["lambda_emotion"],
        lambda_sarcasm=config["lambda_sarcasm"]
    )
    
    # 3. Optimizer & Scheduler
    epochs = config["epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    
    # We will set up optimizer with two param groups later (Phase 2),
    # but initially we freeze the encoder if Phase 1.
    model.freeze_encoder()
    
    # Let's create the parameter groups: encoder and heads
    encoder_params = [p for n, p in model.named_parameters() if "encoder" in n]
    head_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    
    optimizer = AdamW([
        {"params": encoder_params, "lr": config["lr_encoder"]},
        {"params": head_params, "lr": config["lr_heads"]}
    ], weight_decay=0.01)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    best_overall_score = 0.0
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    # History for notebook
    history = []
    
    for epoch in range(1, epochs + 1):
        # Phase handling
        if epoch <= config["freeze_epochs"]:
            model.freeze_encoder()
            print(f"\n--- Epoch {epoch}/{epochs} (Phase 1: Encoder Frozen) ---")
        else:
            model.unfreeze_encoder()
            print(f"\n--- Epoch {epoch}/{epochs} (Phase 2: Full Fine-tuning) ---")
            
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss_dict = criterion(outputs, batch)
                loss = loss_dict["total"]
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            if step % 100 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        avg_train_loss = train_loss / len(train_loader)
        
        # -- EVAL --
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_batches = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=batch_dev["input_ids"], attention_mask=batch_dev["attention_mask"])
                    loss_dict = criterion(outputs, batch_dev)
                    val_loss += loss_dict["total"].item()
                    
                # Store for metrics (move to CPU to save GPU RAM)
                batch_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
                
                all_batches.append(batch_cpu)
                all_outputs.append(out_cpu)
                
        avg_val_loss = val_loss / len(val_loader)
        metrics = compute_metrics(all_outputs, all_batches)
        
        # Summary log
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        df_metrics = pd.DataFrame([{
            "Task": "Sentiment", "F1 Macro": metrics["sentiment_f1_macro"], "Accuracy (or P/R/HL)": metrics["sentiment_accuracy"]
        }, {
            "Task": "Emotion", "F1 Macro": metrics["emotion_f1_macro"], "Accuracy (or P/R/HL)": metrics["emotion_hamming_loss"] 
        }, {
            "Task": "Sarcasm", "F1 Macro": metrics["sarcasm_f1_macro"], "Accuracy (or P/R/HL)": metrics["sarcasm_precision"]
        }])
        print(df_metrics.to_markdown(index=False))
        print(f"\n  Overall Score (Macro-F1 Avg): {metrics['overall_score']:.4f}")
        print(f"  ECE: {metrics['ece']:.4f}")
        
        # Save history
        epoch_hist = {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
        epoch_hist.update(metrics)
        history.append(epoch_hist)
        
        # Checkpoint
        if metrics["overall_score"] > best_overall_score:
            best_overall_score = metrics["overall_score"]
            ckpt_path = os.path.join(config["checkpoint_dir"], f"{config['experiment_name']}_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [*] Saved new best model to {ckpt_path}")
            
    # Save training history
    hist_path = os.path.join(config["checkpoint_dir"], f"{config['experiment_name']}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=4)
        
    print("\nTraining complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_dict = json.load(f)
        
    train(config_dict)
