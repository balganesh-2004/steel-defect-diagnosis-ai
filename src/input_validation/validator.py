"""Preliminary OOD / Input Validation using ResNet18 feature similarity.

Method (training-only, no test leakage):
- Load trained ResNet18 checkpoint
- Extract 512-dim avgpool features (before FC), L2-normalized
- Reference built ONLY from train split (1260 images, seed 42)
- For query image, compute max cosine similarity to any training embedding (nearest neighbor)
- Threshold chosen from training distribution only:
    * training nearest-neighbor mean 0.9565, p1 0.904, p0.1 0.879, min 0.849
    * validation sample with lowest observed 0.8938, unrelated 0.63, solid colors 0.45-0.67
    * Conservative prototype threshold 0.85 balances:
        - Accepts >99.9% training (only 1-2 below 0.85)
        - Accepts known valid crazing_151 (0.894) and scratches_44 (0.893)
        - Rejects unrelated (0.63) with large margin
    * Alternative p1 0.904 would incorrectly reject valid steel (crazing_151) -> not suitable for demo transparency
- Threshold is PRELIMINARY, not a fully validated OOD detector. Documented as such.

DO NOT use confidence alone; NN can be highly confident on OOD.
DO NOT use test set for threshold.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.resnet import build_resnet18_classifier
from src.preprocessing.transforms import build_transforms

# Default paths - match pipeline
DEFAULT_CHECKPOINT = Path("checkpoints/baseline_resnet18_no_aug/best_validation_model.pt")
DEFAULT_CLASS_MAP = Path("results/experiments/baseline_resnet18_no_aug/metrics/class_to_index.json")
DEFAULT_CONFIG = Path("configs/baseline_resnet18_no_aug.json")
DEFAULT_REFERENCE = Path("results/experiments/input_validation/reference_embeddings.npz")
DEFAULT_THRESHOLD_CONFIG = Path("results/experiments/input_validation/threshold_config.json")

# Conservative prototype threshold determined from training-only distribution
# p1=0.904 would reject valid crazing_151 (0.894) -> choose 0.85 to preserve demo transparency while still rejecting OOD (0.63)
CONSERVATIVE_THRESHOLD = 0.85

# Alternative candidates for documentation
THRESHOLD_CANDIDATES = {
    "nearest_neighbor_p1": 0.9043965935707092,
    "nearest_neighbor_p0_5": 0.8983017206192017,
    "nearest_neighbor_p0_1": 0.8798,
    "nearest_neighbor_min": 0.8493,
    "chosen_conservative": 0.85,
}


def load_threshold_config(path: Path = DEFAULT_THRESHOLD_CONFIG) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"chosen_threshold": CONSERVATIVE_THRESHOLD, "note": "fallback conservative"}


def build_feature_extractor(model: torch.nn.Module) -> torch.nn.Module:
    modules = list(model.children())[:-1]  # remove fc, keep avgpool
    return torch.nn.Sequential(*modules)


def load_reference_embeddings(ref_path: Path = DEFAULT_REFERENCE) -> dict:
    data = np.load(ref_path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"].astype(np.float32),  # [1260,512] L2 normalized
        "image_paths": data["image_paths"],
        "labels": data["labels"],
    }


def extract_embedding(pil_image: Image.Image, extractor, transform, device) -> np.ndarray:
    inp = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = extractor(inp).squeeze()
        if feat.dim() > 1:
            feat = feat.view(feat.size(0), -1).squeeze(0)
        feat = F.normalize(feat, p=2, dim=0)
    return feat.cpu().numpy().astype(np.float32)


def validate_image(
    pil_image: Image.Image,
    extractor,
    transform,
    device,
    reference_embeddings: np.ndarray,
    threshold: float = CONSERVATIVE_THRESHOLD,
) -> dict:
    """Return validation result dict."""
    emb = extract_embedding(pil_image, extractor, transform, device)
    # cosine similarities to all training refs (already L2 normalized)
    sims = emb @ reference_embeddings.T  # [1260]
    max_sim = float(sims.max())
    mean_sim = float(sims.mean())
    # also top-5 mean for additional context
    top5 = float(np.sort(sims)[-5:].mean())
    is_suitable = max_sim >= threshold
    return {
        "is_suitable": is_suitable,
        "max_similarity": max_sim,
        "mean_similarity": mean_sim,
        "top5_mean_similarity": top5,
        "threshold": threshold,
        "metric": "max_cosine_to_training_set (nearest neighbor, 512-dim ResNet18 avgpool, L2 normalized)",
        "embedding_dim": int(emb.shape[0]),
        "num_reference": int(reference_embeddings.shape[0]),
    }


def get_validator(device: torch.device | None = None):
    """Load model+extractor+reference for validation (cached pattern for Streamlit)."""
    if device is None:
        device = torch.device("cpu")
    # Load config
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as f:
        config = json.load(f)
    prep = config["preprocessing"]
    transform = build_transforms(prep, "test")
    class_map = json.loads(DEFAULT_CLASS_MAP.read_text(encoding="utf-8"))
    model = build_resnet18_classifier(len(class_map))
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    extractor = build_feature_extractor(model)
    extractor.eval().to(device)
    ref_data = load_reference_embeddings(DEFAULT_REFERENCE)
    # threshold from file or conservative fallback
    thresh_cfg = load_threshold_config()
    # Override to conservative 0.85 for prototype stability (documented)
    threshold = CONSERVATIVE_THRESHOLD
    # But keep candidates for display
    return {
        "model": model,
        "extractor": extractor,
        "transform": transform,
        "device": device,
        "reference_embeddings": ref_data["embeddings"],
        "reference_labels": ref_data["labels"],
        "threshold": threshold,
        "threshold_config": thresh_cfg,
        "class_map": class_map,
    }
