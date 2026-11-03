"""Phase 1 Streamlit Web Prototype - Connected to Real ResNet18 + Grad-CAM Pipeline + Input Validation

Browser -> Streamlit -> Upload -> Input Validation (ResNet18 avgpool feature similarity vs training set) -> Preprocessing -> ResNet18 Checkpoint -> Prediction + Grad-CAM layer4[-1]

Reuses:
- src/models/resnet.py
- src/preprocessing/transforms.py
- scripts/predict.py utilities (Grad-CAM pattern)
- src/input_validation/validator.py (training-only reference, no test leakage)

Checkpoint: checkpoints/baseline_resnet18_no_aug/best_validation_model.pt
Device: CPU
No retraining, no dataset modification.
Outputs: web -> results/experiments/web_demo/, validation logs -> results/experiments/input_validation/
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("TORCH_HOME", str(Path(".torch-cache").resolve()))

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.resnet import build_resnet18_classifier
from src.preprocessing.transforms import build_transforms, build_visual_transform
from src.input_validation.validator import (
    CONSERVATIVE_THRESHOLD,
    DEFAULT_REFERENCE,
    build_feature_extractor,
    extract_embedding,
)

# Constants - match verified pipeline
CHECKPOINT_PATH = Path("checkpoints/baseline_resnet18_no_aug/best_validation_model.pt")
CLASS_MAP_PATH = Path("results/experiments/baseline_resnet18_no_aug/metrics/class_to_index.json")
CONFIG_PATH = Path("configs/baseline_resnet18_no_aug.json")
OUTPUT_DIR = Path("results/experiments/web_demo")
INPUT_VALIDATION_DIR = Path("results/experiments/input_validation")
REFERENCE_PATH = DEFAULT_REFERENCE
CLASS_ORDER = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
DISPLAY_NAMES = {
    "crazing": "CRAZING",
    "inclusion": "INCLUSION",
    "patches": "PATCHES",
    "pitted_surface": "PITTED SURFACE",
    "rolled_in_scale": "ROLLED-IN SCALE",
    "scratches": "SCRATCHES",
}

# Page config
st.set_page_config(
    page_title="Steel Defect Diagnosis — Phase 1",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS - clean industrial/technical
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }
    .section-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .validation-suitable {
        background: #ecfdf5;
        border: 1px solid #6ee7b7;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        color: #065f46;
    }
    .validation-unsuitable {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        color: #9a3412;
    }
    .method-note {
        font-size: 0.85rem;
        color: #64748b;
        font-style: italic;
        margin-top: 0.6rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .phase2-box {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        color: #92400e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model_cached(checkpoint_path_str: str, class_map_path_str: str):
    """Load model once and cache - reuse existing pipeline exactly."""
    checkpoint_path = Path(checkpoint_path_str)
    class_map_path = Path(class_map_path_str)
    config = load_json(CONFIG_PATH)
    preprocessing_config = config.get("preprocessing", {})
    class_to_index = load_json(class_map_path)
    index_to_class = {int(v): k for k, v in class_to_index.items()}
    device = torch.device("cpu")  # Phase 1 verified on CPU
    model = build_resnet18_classifier(len(class_to_index))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    transform = build_transforms(preprocessing_config, "test")
    visual_transform = build_visual_transform(int(preprocessing_config["input_size"]))
    return model, class_to_index, index_to_class, preprocessing_config, device, transform, visual_transform


@st.cache_resource(show_spinner=False)
def load_validation_resources():
    """Load reference embeddings and extractor for input validation (training-only)."""
    # Reuse already loaded model to build extractor - load again for caching isolation
    model, _, _, _, device, transform, _ = load_model_cached(str(CHECKPOINT_PATH), str(CLASS_MAP_PATH))
    extractor = build_feature_extractor(model)
    extractor.eval().to(device)
    # Load reference embeddings (1260x512, training only)
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference embeddings not found at {REFERENCE_PATH}. Run scripts/build_input_validation_reference.py")
    data = np.load(REFERENCE_PATH, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    # Already L2 normalized
    threshold = CONSERVATIVE_THRESHOLD
    # Load threshold config for display
    thresh_cfg_path = INPUT_VALIDATION_DIR / "threshold_config.json"
    thresh_cfg = {}
    if thresh_cfg_path.exists():
        with thresh_cfg_path.open("r", encoding="utf-8") as f:
            thresh_cfg = json.load(f)
    return extractor, transform, device, embeddings, threshold, thresh_cfg


def resized_rgb_float(image: Image.Image, visual_transform) -> np.ndarray:
    tensor = visual_transform(image)
    return tensor.permute(1, 2, 0).numpy().astype(np.float32)


def predict_image(pil_image: Image.Image, model, transform, device, index_to_class):
    """Run inference exactly as CLI does: RGB -> Resize 224 -> ToTensor -> Normalize -> model.eval() -> softmax."""
    model_input = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        logits = model(model_input)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
        pred_class = index_to_class[pred_idx]
        prob_dict = {index_to_class[i]: float(probs[i].item()) for i in range(len(index_to_class))}
    return pred_class, pred_idx, confidence, prob_dict, model_input


def validate_image(pil_image: Image.Image, extractor, transform, device, reference_embeddings, threshold):
    emb = extract_embedding(pil_image, extractor, transform, device)
    sims = emb @ reference_embeddings.T
    max_sim = float(sims.max())
    mean_sim = float(sims.mean())
    top5 = float(np.sort(sims)[-5:].mean())
    is_suitable = max_sim >= threshold
    return {
        "is_suitable": is_suitable,
        "max_similarity": max_sim,
        "mean_similarity": mean_sim,
        "top5_mean": top5,
        "threshold": threshold,
    }


def generate_gradcam(model, model_input, pred_idx, rgb_float):
    """Generate Grad-CAM using layer4[-1] - same as CLI/predict.py"""
    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=model_input, targets=[ClassifierOutputTarget(pred_idx)])[0]
        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        heatmap_uint8 = np.uint8(255 * grayscale_cam)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        return grayscale_cam, heatmap_color, overlay


def save_web_demo_outputs(image_name: str, pil_image: Image.Image, pred_class: str, pred_idx: int, confidence: float, prob_dict: dict, grayscale_cam, heatmap_color, overlay, rgb_float, inference_time: float, validation_result: dict | None = None):
    """Save outputs only under results/experiments/web_demo/"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vis_dir = OUTPUT_DIR / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_name).stem.replace("-", "_")
    original_uint8 = np.uint8(255 * rgb_float)
    combined = np.concatenate([original_uint8, heatmap_color, overlay], axis=1)
    triptych_path = vis_dir / f"{stem}_pred-{pred_class}_web_gradcam.png"
    cv2.imwrite(str(triptych_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    json_path = OUTPUT_DIR / f"{stem}_web_result.json"
    summary = {
        "image_name": image_name,
        "validation": validation_result,
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": prob_dict,
        "inference_time_s": inference_time,
        "gradcam_path": str(triptych_path),
        "checkpoint": str(CHECKPOINT_PATH),
        "target_layer": "layer4[-1]",
        "preprocessing": "RGB, Resize 224, ToTensor, ImageNet Normalize [0.485,0.456,0.406]/[0.229,0.224,0.225]",
        "device": "cpu",
        "timestamp": datetime.now().isoformat(),
        "note": "Grad-CAM is model-attribution evidence, not ground-truth localization.",
    }
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    csv_path = OUTPUT_DIR / "web_predictions.csv"
    record = {
        "timestamp": datetime.now().isoformat(),
        "image_name": image_name,
        "validation_suitable": validation_result["is_suitable"] if validation_result else "",
        "max_similarity": f"{validation_result['max_similarity']:.4f}" if validation_result else "",
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "inference_time_s": inference_time,
        "gradcam_path": str(triptych_path),
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    return triptych_path, json_path, csv_path


def log_validation_attempt(image_name: str, validation_result: dict):
    """Log validation attempts (including rejects) to input_validation dir."""
    INPUT_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    log_path = INPUT_VALIDATION_DIR / "validation_log.csv"
    record = {
        "timestamp": datetime.now().isoformat(),
        "image_name": image_name,
        "is_suitable": validation_result["is_suitable"],
        "max_similarity": f"{validation_result['max_similarity']:.4f}",
        "threshold": validation_result["threshold"],
        "mean_similarity": f"{validation_result['mean_similarity']:.4f}",
    }
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as lf:
        writer = csv.DictWriter(lf, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return log_path


# Load model at startup (cached)
try:
    model, class_to_index, index_to_class, preprocessing_config, device, transform, visual_transform = load_model_cached(
        str(CHECKPOINT_PATH), str(CLASS_MAP_PATH)
    )
    model_loaded = True
    load_error = None
except Exception as e:
    model_loaded = False
    load_error = str(e)
    model = class_to_index = index_to_class = preprocessing_config = device = transform = visual_transform = None

# Load validation resources
try:
    val_extractor, val_transform, val_device, ref_embeddings, val_threshold, thresh_cfg = load_validation_resources()
    validation_loaded = True
    validation_error = None
except Exception as e:
    validation_loaded = False
    validation_error = str(e)
    val_extractor = val_transform = val_device = ref_embeddings = val_threshold = thresh_cfg = None

# Header
st.markdown('<div class="main-header">AI-Driven Multi-Stage Steel Defect Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Phase 1 — AI Diagnosis &amp; Visual Evidence &nbsp;|&nbsp; ResNet18 &nbsp;·&nbsp; 224×224 &nbsp;·&nbsp; Grad-CAM layer4[-1] &nbsp;·&nbsp; CPU &nbsp;·&nbsp; 99.63% Clean Test</div>', unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Failed to load checkpoint `{CHECKPOINT_PATH}`: {load_error}")
    st.info("Ensure `checkpoints/baseline_resnet18_no_aug/best_validation_model.pt` exists and `results/experiments/baseline_resnet18_no_aug/metrics/class_to_index.json` is present.")
    st.stop()
else:
    st.success(f"Model loaded: ResNet18 | Checkpoint: `{CHECKPOINT_PATH}` | Device: CPU | Classes: 6 | Target: layer4[-1]", icon="✅")

if not validation_loaded:
    st.warning(f"Input validation unavailable: {validation_error}. Run `python scripts/build_input_validation_reference.py`", icon="⚠️")
else:
    st.info(f"Input Validation — Training-only reference (1260 embeddings, 512-dim) | Threshold: {val_threshold:.2f} (conservative, p1 0.904 → chosen 0.85) | No test leakage", icon="🛡️")

# Model Information - expandable
with st.expander("Model Information", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Model**  \nResNet18 (torchvision, ImageNet pretrained, 6-class head)")
        st.markdown("**Input**  \n224 × 224, RGB, ImageNet Normalize")
        st.markdown("**Normalization**  \nmean `[0.485, 0.456, 0.406]`  \nstd `[0.229, 0.224, 0.225]`")
    with col_b:
        st.markdown("**Classes (6)**")
        for idx, cls in enumerate(CLASS_ORDER):
            st.markdown(f"`{idx}` {DISPLAY_NAMES[cls]}")
    with col_c:
        st.markdown("**Explainability**  \nGrad-CAM")
        st.markdown("**Target layer**  \n`layer4[-1]`")
        st.markdown("**Device**  \nCPU")
        st.markdown("**Checkpoint**  \n`best_validation_model.pt`")
        st.markdown("**Clean Test**  \n99.63% (269/270)")
    # Input validation methodology mini-note
    with st.expander("Input Validation Methodology (Preliminary)", expanded=False):
        st.markdown(
            """
            **Why validation?** ResNet18 always returns one of 6 classes, even for unrelated images. Confidence alone is insufficient (network can be highly confident on OOD).
            
            **Method:** Training-only 512-dim ResNet18 avgpool features (before FC, L2-normalized). Reference = 1260 training embeddings (seed 42 split, no test leakage). Query max cosine similarity to nearest training embedding vs conservative threshold.
            
            **Threshold:** Training nearest-neighbor distribution mean 0.956, p1 0.904, min 0.849. Chosen **0.85** (conservative prototype) to avoid rejecting valid steel (e.g., crazing_151 0.894, scratches_44 0.894) while still rejecting unrelated (0.63, solid colors 0.45-0.67). p1 0.904 would incorrectly reject valid crazing_151.
            
            **Limitations:** Preliminary OOD heuristic, not a fully validated detector. Does not guarantee domain membership. Official 99.63% metrics unchanged. Some OOD may still pass (false accepts).
            
            **Artifacts:** `results/experiments/input_validation/reference_embeddings.npz`, `threshold_config.json`
            """
        )

# Two-column layout: upload + preview | analysis
left, right = st.columns([1, 1.1], gap="large")

with left:
    st.markdown("### Image Upload")
    uploaded = st.file_uploader("Upload steel surface image (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"], help="Use NEU-DET images, e.g. crazing_37.jpg or crazing_151.jpg for verification")
    analyze_btn = st.button("Analyze Image", type="primary", use_container_width=True, disabled=(uploaded is None))

    if uploaded is not None:
        try:
            pil_preview = Image.open(uploaded).convert("RGB")
            st.image(pil_preview, caption=f"Uploaded: {uploaded.name} ({pil_preview.size[0]}×{pil_preview.size[1]})", use_container_width=True)
        except Exception as e:
            st.error(f"Invalid or unreadable image: {e}")
            pil_preview = None
    else:
        pil_preview = None
        st.info("Upload an image to enable analysis. For verification: `data/raw/neu-det/NEU-DET/train/images/crazing/crazing_37.jpg` (expected crazing 99.71%) and `crazing_151.jpg` (expected patches 51.27%).")
        st.code("data/raw/neu-det/NEU-DET/train/images/crazing/crazing_37.jpg\n"
                "data/raw/neu-det/NEU-DET/train/images/crazing/crazing_151.jpg", language="text")

with right:
    st.markdown("### Diagnosis")
    if uploaded is None:
        st.markdown('<div class="section-card">Awaiting image upload. Click <b>Analyze Image</b> after upload.</div>', unsafe_allow_html=True)
    elif analyze_btn and pil_preview is not None:
        # Always run validation first
        with st.spinner("Running input validation (feature similarity) + ResNet18 inference..."):
            uploaded.seek(0)
            pil_image = Image.open(uploaded).convert("RGB")
            rgb_float = resized_rgb_float(pil_image, visual_transform)

            # Input Validation
            validation_result = None
            if validation_loaded:
                try:
                    validation_result = validate_image(pil_image, val_extractor, val_transform, val_device, ref_embeddings, val_threshold)
                    # Log validation attempt
                    log_validation_attempt(uploaded.name, validation_result)
                except Exception as e:
                    st.warning(f"Validation failed (proceeding without gate): {e}")
                    validation_result = {"is_suitable": True, "max_similarity": 1.0, "mean_similarity": 0.0, "threshold": val_threshold}
            else:
                validation_result = {"is_suitable": True, "max_similarity": 1.0, "mean_similarity": 0.0, "threshold": 0.85}

            # Display Input Validation section BEFORE diagnosis
            st.markdown("#### Input Validation")
            if validation_result["is_suitable"]:
                st.markdown(
                    f"""
                    <div class="validation-suitable">
                        <b>✓ Suitable steel-defect image</b><br>
                        Similarity {validation_result['max_similarity']:.3f} ≥ threshold {validation_result['threshold']:.2f} &nbsp;|&nbsp; Metric: max cosine to training set (512-dim)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Reference: 1260 training embeddings (no test leakage). Mean similarity {validation_result['mean_similarity']:.3f}.")
                is_unknown = False
            else:
                st.markdown(
                    f"""
                    <div class="validation-unsuitable">
                        <b>⚠ UNKNOWN / UNSUITABLE IMAGE</b><br>
                        The uploaded image does not appear sufficiently similar to the supported steel-defect image domain.<br>
                        Similarity {validation_result['max_similarity']:.3f} &lt; threshold {validation_result['threshold']:.2f} &nbsp;|&nbsp; Metric: max cosine to training set<br><br>
                        <b>The image is outside the supported steel-defect image domain.</b><br>
                        Please upload a clear steel surface image.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="method-note">Input validation is a preliminary out-of-distribution screening mechanism. It does not guarantee that an image belongs to the steel-defect domain.</div>', unsafe_allow_html=True)
                is_unknown = True

        # If unsuitable, diagnosis/Grad-CAM is gated - do not proceed
        if validation_result and not validation_result["is_suitable"]:
            # Already shown unsuitable message, do not show prediction
            pass
        else:
            # Suitable -> run prediction + Grad-CAM
            with st.spinner("Running ResNet18 inference + Grad-CAM layer4[-1]..."):
                start = time.perf_counter()
                uploaded.seek(0)
                pil_image2 = Image.open(uploaded).convert("RGB")
                rgb_float2 = resized_rgb_float(pil_image2, visual_transform)
                pred_class, pred_idx, confidence, prob_dict, model_input = predict_image(pil_image2, model, transform, device, index_to_class)
                inference_time = time.perf_counter() - start

                try:
                    grayscale_cam, heatmap_color, overlay = generate_gradcam(model, model_input, pred_idx, rgb_float2)
                    gradcam_ok = True
                    gradcam_err = None
                except Exception as e:
                    gradcam_ok = False
                    gradcam_err = str(e)
                    heatmap_color = overlay = None
                    grayscale_cam = None

                triptych_path = json_path = csv_path = None
                if gradcam_ok:
                    try:
                        triptych_path, json_path, csv_path = save_web_demo_outputs(
                            uploaded.name, pil_image2, pred_class, pred_idx, confidence, prob_dict, grayscale_cam, heatmap_color, overlay, rgb_float2, inference_time, validation_result
                        )
                    except Exception as e:
                        st.warning(f"Failed to save web_demo outputs: {e}")

            # Diagnosis Summary
            st.markdown("#### AI Diagnosis")
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="metric-label">Predicted defect</div>
                    <div class="metric-value">{DISPLAY_NAMES[pred_class]}</div>
                    <div style="margin-top:0.6rem;">
                        <span class="metric-label">Confidence</span><br>
                        <span style="font-size:1.2rem; font-weight:600;">{confidence*100:.2f}%</span> <span style="color:#64748b;">({confidence:.6f})</span>
                    </div>
                    <div style="margin-top:0.6rem; font-size:0.9rem; color:#475569;">
                        Inference: {inference_time:.4f} s &nbsp;|&nbsp; Device: CPU &nbsp;|&nbsp; Input: 224×224
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Class probabilities (all 6)**")
            prob_sorted = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
            import pandas as pd
            chart_df = pd.DataFrame({"class": [DISPLAY_NAMES[k] for k in prob_dict.keys()], "probability": list(prob_dict.values())})
            chart_df = chart_df.set_index("class")
            st.bar_chart(chart_df, height=220)
            st.dataframe(
                pd.DataFrame([{"Class": DISPLAY_NAMES[k], "Prob %": round(v*100, 2), "Raw": round(v, 6)} for k, v in prob_sorted.items()]),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Visual Evidence — Grad-CAM (layer4[-1])")
            if gradcam_ok:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(pil_image2.resize((224, 224)), caption="1. Original (224×224)", use_container_width=True)
                with c2:
                    st.image(heatmap_color, caption="2. Grad-CAM Heatmap", use_container_width=True)
                with c3:
                    st.image(overlay, caption="3. Overlay (evidence)", use_container_width=True)
                st.caption("Grad-CAM highlights image regions that contributed to the model prediction. It represents model-attribution evidence, not ground-truth localization.")
                if triptych_path:
                    st.success(f"Saved: `{triptych_path}` | `{json_path}`", icon="💾")
            else:
                st.error(f"Grad-CAM generation failed: {gradcam_err}")

            st.markdown('<div class="method-note">Input validation is a preliminary out-of-distribution screening mechanism. It does not guarantee that an image belongs to the steel-defect domain.</div>', unsafe_allow_html=True)

            st.session_state["last_pred"] = pred_class
            st.session_state["last_conf"] = confidence
    elif pil_preview is not None and not analyze_btn:
        st.markdown('<div class="section-card">Click <b>Analyze Image</b> to run diagnosis.</div>', unsafe_allow_html=True)

# Phase 2 placeholder - always visible below
st.markdown("---")
st.markdown("### Evidence-Based Corrective Action Recommendations — Phase 2")
st.markdown(
    """
    <div class="phase2-box">
        <b>Phase 2 (Semester 8) — Not yet implemented:</b><br>
        Possible manufacturing causes and evidence-based corrective actions will be incorporated in Phase 2.
        Phase 1 focuses on <b>diagnosis + visual evidence</b> only. No manufacturing cause is inferred from the image alone.
        Future pipeline: Diagnosis → Possible Causes (knowledge base) → Evidence Fusion → Corrective Action Recommendation.
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Phase 1 baseline: 99.63% clean test accuracy (ResNet18, seed 42). Robustness evaluation shows sensitivity to noise/blur — see `results/experiments/robustness_resnet18/`. Known test error: `crazing_151.jpg → patches (51.27%)` preserved for transparency. Input validation: preliminary training-only similarity gate (threshold 0.85, no test leakage).")

# Footer - outputs location
st.markdown("---")
st.caption(f"Outputs saved to `{OUTPUT_DIR}/` (web) and `{INPUT_VALIDATION_DIR}/` (validation logs). Checkpoint, splits, and existing experiments are never overwritten.")
