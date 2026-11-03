"""Build training-only reference embeddings for input validation / OOD screening.

Uses existing ResNet18 checkpoint and extracts features before final FC (512-dim).
Reference built ONLY from training split (seed_42_70_15_15/train.csv) - no test leakage.
Saves to results/experiments/input_validation/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("TORCH_HOME", str(Path(".torch-cache").resolve()))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet import build_resnet18_classifier
from src.preprocessing.transforms import build_transforms

CHECKPOINT = Path("checks/checkpoints/baseline_resnet18_no_aug/best_validation_model.pt")
# Correct path
CHECKPOINT = Path("checkpoints/baseline_resnet18_no_aug/best_validation_model.pt")
CLASS_MAP = Path("results/experiments/baseline_resnet18_no_aug/metrics/class_to_index.json")
CONFIG = Path("configs/baseline_resnet18_no_aug.json")
TRAIN_CSV = Path("data/splits/seed_42_70_15_15/train.csv")
OUTPUT_DIR = Path("results/experiments/input_validation")


def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_feature_extractor(model: torch.nn.Module) -> torch.nn.Module:
    """Return model that outputs 512-dim avgpool features (before FC)."""
    # ResNet children: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
    # Remove fc
    modules = list(model.children())[:-1]  # up to avgpool
    extractor = torch.nn.Sequential(*modules)
    return extractor


def extract_embedding(pil_image: Image.Image, extractor, transform, device) -> np.ndarray:
    inp = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = extractor(inp)  # [1,512,1,1]
        feat = feat.squeeze()  # [512] or [512,1,1]
        if feat.dim() > 1:
            feat = feat.view(feat.size(0), -1).squeeze(0)
        # L2 normalize
        feat = F.normalize(feat, p=2, dim=0)
    return feat.cpu().numpy().astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_json(CONFIG)
    prep = config["preprocessing"]
    transform = build_transforms(prep, "test")
    class_to_index = load_json(CLASS_MAP)
    index_to_class = {int(v): k for k, v in class_to_index.items()}

    device = torch.device("cpu")
    model = build_resnet18_classifier(len(class_to_index))
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    extractor = build_feature_extractor(model)
    extractor.eval()
    extractor.to(device)

    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Training manifest: {len(train_df)} images")
    print(f"Per class: {train_df['class'].value_counts().to_dict()}")
    # Normalize class names
    train_df["class"] = train_df["class"].replace({"rolled-in_scale": "rolled_in_scale"})

    embeddings = []
    image_paths = []
    labels = []
    failed = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting train embeddings"):
        img_path = Path(row["image_path"])
        # Handle Windows vs stored path case: stored is data/raw/NEU-DET/... but actual may be neu-det
        # Try as stored, then fallback to lower
        if not img_path.exists():
            alt = Path(str(img_path).replace("data/raw/NEU-DET", "data/raw/neu-det"))
            if alt.exists():
                img_path = alt
            else:
                failed.append(str(row["image_path"]))
                continue
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                emb = extract_embedding(im, extractor, transform, device)
                embeddings.append(emb)
                image_paths.append(str(row["image_path"]))
                labels.append(row["class"])
        except Exception as e:
            print(f"Failed {img_path}: {e}")
            failed.append(str(img_path))

    embeddings = np.stack(embeddings)  # [1260,512]
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Failed: {len(failed)}")

    # Compute class centroids
    centroids = {}
    for cls in sorted(set(labels)):
        cls_embs = embeddings[np.array(labels) == cls]
        centroid = cls_embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[cls] = centroid

    # Save reference
    npz_path = OUTPUT_DIR / "reference_embeddings.npz"
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        image_paths=np.array(image_paths),
        labels=np.array(labels),
        class_order=np.array(sorted(set(labels))),
    )
    # Save centroids separately
    centroid_path = OUTPUT_DIR / "class_centroids.npz"
    np.savez_compressed(centroid_path, **centroids)

    print(f"Saved {npz_path}")
    print(f"Saved {centroid_path}")

    # Analyze self-similarity distribution for threshold
    # For each embedding, compute max cosine similarity to other training embeddings (excluding self)
    # This is O(N^2) but N=1260, 512 dim => feasible with matrix multiplication
    # embeddings are already L2 normalized, so cosine = dot
    print("Computing leave-one-out max similarities...")
    # Compute similarity matrix: [1260,1260]
    sim_matrix = embeddings @ embeddings.T  # cosine
    # Set diagonal to -1 to exclude self
    np.fill_diagonal(sim_matrix, -1.0)
    max_sims = sim_matrix.max(axis=1)  # [1260]
    # Also compute similarity to nearest centroid
    centroid_matrix = np.stack([centroids[c] for c in sorted(centroids.keys())])  # [6,512]
    centroid_sims = embeddings @ centroid_matrix.T  # [1260,6]
    max_centroid_sims = centroid_sims.max(axis=1)

    # Statistics
    def stats(arr, name):
        print(f"\n{name} stats:")
        print(f"  mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}")
        for p in [0.1, 1, 5, 10, 25, 50, 90, 95, 99, 99.5, 99.9]:
            print(f"  p{p:4.1f}={np.percentile(arr, p):.4f}")

    stats(max_sims, "Max similarity to other train embedding (nearest neighbor)")
    stats(max_centroid_sims, "Max similarity to class centroid")

    # Suggest conservative threshold: Use 1st percentile of max_sims
    # This means 99% of training images would be accepted
    # Also consider mean -3*std
    thresh_p1 = np.percentile(max_sims, 1)
    thresh_p05 = np.percentile(max_sims, 0.5)
    thresh_mean3std = max_sims.mean() - 3 * max_sims.std()
    thresh_p1_cent = np.percentile(max_centroid_sims, 1)
    print("\nSuggested thresholds (training-only, no test leakage):")
    print(f"  nearest-neighbor p1 (1st percentile) = {thresh_p1:.4f}")
    print(f"  nearest-neighbor p0.5 = {thresh_p05:.4f}")
    print(f"  nearest-neighbor mean-3std = {thresh_mean3std:.4f}")
    print(f"  centroid p1 = {thresh_p1_cent:.4f}")

    # Choose conservative threshold: use p1 of nearest-neighbor, but also ensure not too strict
    # For prototype, we will use 0.85 or p1 whichever is more conservative (lower)
    # Let's record all and choose p1 as default but document
    config_out = {
        "reference_source": str(TRAIN_CSV),
        "num_reference": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "extractor": "ResNet18 avgpool (512-dim) before FC, L2 normalized",
        "preprocessing": prep,
        "checkpoint": str(CHECKPOINT),
        "threshold_method": "1st percentile of training nearest-neighbor max cosine similarity (no test leakage)",
        "threshold_candidates": {
            "nearest_neighbor_p1": float(thresh_p1),
            "nearest_neighbor_p0_5": float(thresh_p05),
            "nearest_neighbor_mean_minus_3std": float(thresh_mean3std),
            "centroid_p1": float(thresh_p1_cent),
        },
        "chosen_threshold": float(thresh_p1),
        "chosen_metric": "max_cosine_to_training_set (nearest neighbor)",
        "note": "Preliminary OOD heuristic. Not a fully validated OOD detector. Does not guarantee domain membership. Official 99.63% metrics unchanged. If unreliable, document limitation rather than arbitrary confidence threshold.",
        "stats": {
            "nn_mean": float(max_sims.mean()),
            "nn_std": float(max_sims.std()),
            "nn_min": float(max_sims.min()),
            "nn_max": float(max_sims.max()),
            "centroid_mean": float(max_centroid_sims.mean()),
        },
    }
    # Also compute histogram for documentation
    with (OUTPUT_DIR / "threshold_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)
    print(f"\nSaved threshold config to {OUTPUT_DIR / 'threshold_config.json'}")

    # Also test on a few unrelated dummy embeddings to see separation
    # Create a random noise image embedding as quick sanity check
    from PIL import Image as PILImage
    dummy = PILImage.new("RGB", (200, 200), color=(128, 128, 128))
    # Add text-like random - actually use solid color
    dummy_emb = extract_embedding(dummy, extractor, transform, device)
    dummy_sims = dummy_emb @ embeddings.T
    print(f"\nDummy solid gray max similarity to train: {dummy_sims.max():.4f} (should be < threshold if method works)")

    # Try new.png as unrelated
    for test_path in [Path("C:/new.png"), Path("new.png")]:
        if test_path.exists():
            try:
                with PILImage.open(test_path) as im2:
                    im2 = im2.convert("RGB")
                    emb2 = extract_embedding(im2, extractor, transform, device)
                    sims2 = emb2 @ embeddings.T
                    print(f"Unrelated {test_path} max similarity: {sims2.max():.4f}")
            except Exception as e:
                print(f"Failed unrelated {test_path}: {e}")

if __name__ == "__main__":
    main()
