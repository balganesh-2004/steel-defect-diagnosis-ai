"""Test input validation logic on required images."""
from pathlib import Path
import json, sys
sys.path.insert(0, ".")
from PIL import Image
import torch
import numpy as np
from src.input_validation.validator import CONSERVATIVE_THRESHOLD, load_reference_embeddings, build_feature_extractor, extract_embedding
from src.models.resnet import build_resnet18_classifier
from src.preprocessing.transforms import build_transforms
import torch.nn.functional as F

ckpt = Path("checkpoints/baseline_resnet18_no_aug/best_validation_model.pt")
cls_map = json.load(open("results/experiments/baseline_resnet18_no_aug/metrics/class_to_index.json"))
cfg = json.load(open("configs/baseline_resnet18_no_aug.json"))
prep = cfg["preprocessing"]
trans = build_transforms(prep, "test")
device = torch.device("cpu")
model = build_resnet18_classifier(len(cls_map))
model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
model.eval()
extractor = build_feature_extractor(model)
extractor.eval().to(device)
ref = load_reference_embeddings()["embeddings"]
threshold = CONSERVATIVE_THRESHOLD
print(f"Threshold: {threshold}")

def validate_and_predict(img_path_str):
    pil = Image.open(img_path_str).convert("RGB")
    emb = extract_embedding(pil, extractor, trans, device)
    sims = emb @ ref.T
    max_sim = float(sims.max())
    is_suitable = max_sim >= threshold
    # also predict
    inp = trans(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        conf = float(probs[pred_idx].item())
        idx_to_cls = {int(v):k for k,v in cls_map.items()}
        pred_cls = idx_to_cls[pred_idx]
    status = "SUITABLE" if is_suitable else "UNKNOWN/UNSUITABLE"
    print(f"\n{img_path_str}")
    print(f"  Validation: {status}  max_sim={max_sim:.4f} threshold={threshold:.2f}")
    if is_suitable:
        print(f"  Prediction: {pred_cls} confidence {conf:.6f} ({conf*100:.2f}%)")
    else:
        print(f"  Prediction: SKIPPED (would be {pred_cls} {conf:.4f})")
        print(f"  Message: The uploaded image does not appear sufficiently similar to the supported steel-defect image domain.")
    return is_suitable, pred_cls, conf, max_sim

print("=== Testing required cases ===")
validate_and_predict("data/raw/neu-det/NEU-DET/train/images/crazing/crazing_37.jpg")
validate_and_predict("data/raw/neu-det/NEU-DET/train/images/crazing/crazing_151.jpg")
# unrelated
validate_and_predict("C:/new.png")
# extra steel
validate_and_predict("data/raw/neu-det/NEU-DET/train/images/inclusion/inclusion_20.jpg")
validate_and_predict("data/raw/neu-det/NEU-DET/train/images/patches/patches_10.jpg")
# solid color
from PIL import Image as PILI
import tempfile
tmp = Path(tempfile.gettempdir()) / "solid_red_test.png"
PILI.new("RGB",(224,224),color=(255,0,0)).save(tmp)
validate_and_predict(str(tmp))
print("\n=== Done ===")
