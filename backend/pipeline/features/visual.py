from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import VisualFeatures, VisualFrameFeature
from backend.pipeline.logging_setup import get_logger
from backend.pipeline.features.ocr_signals import (
    extract_text as _ocr_extract_text,
    detect_commercial_patterns,
)

# ── Thresholds ────────────────────────────────────────────────────────────────
DEFAULT_CUT_THRESHOLD = 0.6
BLACK_FRAME_LUMINANCE_MAX = 10.0   # mean brightness below this → black frame
BLACK_FRAME_VARIANCE_MAX = 50.0    # variance below this → pure-color frame

# OCR is the most expensive per-frame step. We sample every Nth frame and reuse
# the result on the skipped frames (forward-fill). On-screen text typically
# persists ≥3 s, so stride 3 has negligible recall impact and triples throughput.
OCR_STRIDE_SEC = 3
# Skip OCR on frames whose pixel variance is below this — too uniform to host text.
OCR_MIN_LUMINANCE_VARIANCE = 100.0


def _should_run_ocr(
    frame_idx: int,
    stride: int,
    is_black: bool,
    lum_var: float,
) -> bool:
    """Gating logic for whether to run OCR on a particular frame.

    Returns True only when ALL of:
      - frame_idx is on a stride boundary
      - frame is not black
      - frame variance is high enough to plausibly contain text
    """
    if stride > 1 and (frame_idx % stride) != 0:
        return False
    if is_black:
        return False
    if lum_var < OCR_MIN_LUMINANCE_VARIANCE:
        return False
    return True

# ── CLIP scene prompts (10 labels, covering all taxonomy classes) ─────────────
# Expanded from 4 → 10 to make all taxonomy labels reachable via the CLIP
# fallback classifier in classify.py.
SCENE_LABELS = [
    # Core content
    "a presentation slide",
    "a person talking to a camera",
    "a screen recording of software",
    # Transition / structural
    "a blank screen",
    "an end credits screen",
    "a title card",
    # Sponsor / promo
    "an advertisement slide",
    "a sponsor logo or product",
    # Other non-content
    "a video game interface",
    "an animated scene",
]


# ── Helper functions ──────────────────────────────────────────────────────────

def _load_clip_model(device: str) -> Tuple:
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    return processor, model


def _calculate_histogram(img_bgr: np.ndarray) -> np.ndarray:
    """Normalized 3D HSV histogram for scene-cut detection."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _detect_black_frame(
    gray: np.ndarray,
    lum_max: float = BLACK_FRAME_LUMINANCE_MAX,
    var_max: float = BLACK_FRAME_VARIANCE_MAX,
) -> Tuple[float, float, bool]:
    """Return (mean_luminance, variance, is_black_frame)."""
    mean_lum = float(gray.mean())
    var_lum = float(gray.var())
    return mean_lum, var_lum, (mean_lum < lum_max and var_lum < var_max)


def _compute_motion_intensity(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Mean absolute pixel difference between consecutive grayscale frames."""
    return float(cv2.absdiff(prev_gray, curr_gray).mean())


def _compute_chroma_diff(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    """
    [V2.1] Chroma-channel difference with 4:2:0 downsampling.

    Mirrors the YCbCr 4:2:0 color subsampling used in MPEG/H.264 video
    (CSCI 576 §3.2). By operating on the downsampled chroma channels we
    reduce sensitivity to luma-only changes (lighting shifts) and focus on
    true color content changes — a better signal for scene-cut detection than
    luma alone.
    """
    def to_420_chroma(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        _, cb, cr = cv2.split(ycbcr)
        h, w = cb.shape
        # 2× spatial downsampling in both axes (4:2:0)
        cb_ds = cv2.resize(cb, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        cr_ds = cv2.resize(cr, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        return cb_ds, cr_ds

    cb_p, cr_p = to_420_chroma(prev_bgr)
    cb_c, cr_c = to_420_chroma(curr_bgr)
    cb_diff = cv2.absdiff(cb_p.astype(np.float32), cb_c.astype(np.float32)).mean()
    cr_diff = cv2.absdiff(cr_p.astype(np.float32), cr_c.astype(np.float32)).mean()
    return float((cb_diff + cr_diff) / 2.0 / 255.0)


def _compute_dct_hf_energy(gray: np.ndarray, block_size: int = 8) -> float:
    """
    [V2.1] Normalized high-frequency DCT energy over the frame.

    Divides the frame into 8×8 blocks (identical to JPEG/MPEG block structure
    from CSCI 576 §4) and computes cv2.dct() on each. The bottom-right quadrant
    of each DCT block carries high-frequency coefficients; their summed energy
    indicates texture/detail richness. A title card or blank screen will have
    near-zero HF energy; a busy animated frame will have high energy.
    """
    h, w = gray.shape
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    if h_crop == 0 or w_crop == 0:
        return 0.0
    gray_f = gray[:h_crop, :w_crop].astype(np.float32)

    total_hf = 0.0
    count = 0
    for i in range(0, h_crop, block_size):
        for j in range(0, w_crop, block_size):
            block = gray_f[i : i + block_size, j : j + block_size]
            dct_block = cv2.dct(block)
            # High-frequency region: lower-right 4×4 of the 8×8 block
            hf = dct_block[block_size // 2 :, block_size // 2 :]
            total_hf += float(np.sum(hf ** 2))
            count += 1

    # Normalise by block count and block area squared so the value is scale-invariant
    return total_hf / max(count, 1) / (block_size ** 4)


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_visual_features(
    workspace: Workspace,
    device: str = "cpu",
    cut_threshold: float = DEFAULT_CUT_THRESHOLD,
) -> Path:
    """
    Reads 1 FPS JPEGs from frames_cache/, computes per-frame features, and
    writes features/visual_features.json.

    Per-frame outputs:
    - HSV histogram diff + is_hard_cut
    - CLIP zero-shot scene probabilities (10 labels)
    - Mean luminance, variance, is_black_frame
    - Motion intensity (absolute frame diff)
    - [V2.1] 4:2:0 chroma diff
    - [V2.1] 8×8 DCT high-frequency energy
    """
    log = get_logger(workspace.log_path)
    out_path = workspace.visual_features_path

    if out_path.exists():
        log.info("Visual features already exist. Skipping.")
        return out_path

    frame_files = sorted(workspace.frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        log.warning("No frames found in frames_cache/. Skipping visual features.")
        return out_path

    log.info("Stage 2 (Visual): Loading CLIP model (%s)...", device)
    processor, clip_model = _load_clip_model(device)

    log.info("Processing %d frames...", len(frame_files))

    frame_features = []
    prev_hist: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None
    prev_bgr: Optional[np.ndarray] = None

    # Forward-fill state for OCR results across stride-skipped frames.
    last_ocr_lines: list[str] = []
    last_commercial: dict = {
        "has_url": False, "has_price": False, "has_phone": False,
        "has_cta": False, "has_brand_lockup": False,
    }

    for frame_path in frame_files:
        frame_idx = int(frame_path.stem.split("_")[1])
        timestamp_sec = float(frame_idx)

        # ── Load image ──────────────────────────────────────────────────────
        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            log.warning("Could not read frame %s — skipping.", frame_path.name)
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ── HSV histogram diff ───────────────────────────────────────────────
        curr_hist = _calculate_histogram(img_bgr)
        hist_diff = 0.0
        is_cut = False
        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            hist_diff = max(0.0, 1.0 - similarity)
            is_cut = hist_diff > cut_threshold

        # ── Black-frame detection ────────────────────────────────────────────
        mean_lum, var_lum, is_black = _detect_black_frame(gray)

        # ── Motion intensity ─────────────────────────────────────────────────
        motion = 0.0
        if prev_gray is not None:
            motion = _compute_motion_intensity(prev_gray, gray)

        # ── [V2.1] Chroma diff (4:2:0) ───────────────────────────────────────
        chroma_diff = 0.0
        if prev_bgr is not None:
            chroma_diff = _compute_chroma_diff(prev_bgr, img_bgr)

        # ── [V2.1] DCT high-frequency energy ────────────────────────────────
        dct_energy = _compute_dct_hf_energy(gray)

        # ── CLIP zero-shot scene classification + pooled image embedding ────
        pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        text_inputs = processor(
            text=SCENE_LABELS, return_tensors="pt", padding=True
        ).to(device)
        image_inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_outputs = clip_model.get_image_features(**image_inputs)
            text_outputs = clip_model.get_text_features(**text_inputs)
            # transformers >=5 returns BaseModelOutputWithPooling whose
            # `pooler_output` is the projected (image|text) embedding tensor;
            # earlier versions returned the tensor directly.
            image_features = getattr(image_outputs, "pooler_output", image_outputs)
            text_features = getattr(text_outputs, "pooler_output", text_outputs)
            # Normalise then dot-product = cosine similarity.
            image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = image_norm @ text_norm.T * clip_model.logit_scale.exp()
            probs = logits.softmax(dim=1)[0].cpu().numpy()
            pooled = image_norm[0].cpu().numpy()
        clip_results = {label: float(prob) for label, prob in zip(SCENE_LABELS, probs)}
        clip_embedding = [round(float(x), 6) for x in pooled.tolist()]

        # ── OCR + commercial-pattern detection (with stride + luminance gate)
        if _should_run_ocr(frame_idx, OCR_STRIDE_SEC, is_black, var_lum):
            try:
                last_ocr_lines = _ocr_extract_text(img_bgr, device=device)
            except Exception as e:
                log.warning("OCR failed on frame %d: %s", frame_idx, e)
                last_ocr_lines = []
            last_commercial = detect_commercial_patterns(last_ocr_lines)
        # else: keep last_ocr_lines / last_commercial as-is (forward-fill)
        ocr_joined = " ".join(last_ocr_lines)[:500]

        # ── Assemble feature record ──────────────────────────────────────────
        frame_features.append(
            VisualFrameFeature(
                frame_index=frame_idx,
                timestamp_sec=timestamp_sec,
                hist_diff_to_previous=round(hist_diff, 6),
                is_hard_cut=is_cut,
                clip_labels=clip_results,
                clip_embedding=clip_embedding,
                mean_luminance=round(mean_lum, 3),
                luminance_variance=round(var_lum, 3),
                is_black_frame=is_black,
                motion_intensity=round(motion, 4),
                chroma_diff=round(chroma_diff, 6),
                dct_hf_energy=round(dct_energy, 6),
                ocr_text=ocr_joined,
                has_url=last_commercial["has_url"],
                has_price=last_commercial["has_price"],
                has_phone=last_commercial["has_phone"],
                has_cta=last_commercial["has_cta"],
                has_brand_lockup=last_commercial["has_brand_lockup"],
            )
        )

        prev_hist = curr_hist
        prev_gray = gray
        prev_bgr = img_bgr.copy()

    # ── Release CLIP model before next stage ────────────────────────────────
    del clip_model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = VisualFeatures(frames=frame_features)
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    log.info("Visual features saved to %s (%d frames).", out_path.name, len(frame_features))
    return out_path
