import torch
from torch.utils.data import DataLoader
from dataset import FMADataset, custom_collate_fn
from fusion_model import FusionModel
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_case_studies():
    config = load_config()
    device = torch.device("cpu")
    
    # Load val dataset
    val_dataset = FMADataset("data/splits/val.json", tokenizer_name=config["text"]["model_name"], max_length=config["text"]["max_length"], load_graph=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    # Initialize model (we assume it's untrained or partially trained here, just demonstrating the inference code)
    model = FusionModel(text_model_name=config["text"]["model_name"], num_classes=8, fusion_type="cross_attention").to(device)
    model.eval()
    
    print("========================================")
    print("      Task 3: Qualitative Case Studies")
    print("========================================\n")
    
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            graph_data = batch['graph_data']
            if graph_data[0] == "NONE": continue
            
            graph_data = graph_data[0].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'][0].numpy()
            
            logits, _ = model(input_ids, attention_mask, graph_data)
            probs = torch.sigmoid(logits).squeeze(0).numpy()
            
            # Map back to genres (mock mapping based on FMA top genres)
            genres = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]
            
            true_genres = [genres[j] for j in range(8) if labels[j] == 1.0]
            pred_genres = [genres[j] for j in range(8) if probs[j] > 0.5]
            if not pred_genres:
                pred_genres = [genres[probs.argmax()]]
                
            text = val_dataset.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            print(f"--- Case Study {count+1} ---")
            print(f"Track ID: {batch['track_id'][0]}")
            print(f"Caption: '{text}'")
            print(f"Ground Truth Genres: {true_genres}")
            print(f"Predicted Genres (Prob > 0.5): {pred_genres}")
            print("Probabilities:")
            for j in range(8):
                print(f"  {genres[j]}: {probs[j]:.3f}")
            print("\n")
            
            count += 1
            if count >= 3:
                break

if __name__ == "__main__":
    run_case_studies()
