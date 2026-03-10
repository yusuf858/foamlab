"""
FoamLab — Complete Integrated Detection Pipeline
Assignment-3: Foam/Oil Pollution Detection in Lake Surface Images
Pipeline: Input → Preprocessing → Enhancement → Segmentation →
          Morphological Refinement → Feature Extraction → Classification
"""
import time
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ─── Pollution class definitions ─────────────────────────────────────────────
POLLUTION_CLASSES = {
    "Foam Pollution":      {"color": (255, 80,  80),  "hex": "#FF5050"},
    "Oil Slick":           {"color": (255, 165,  0),  "hex": "#FFA500"},
    "Surface Reflection":  {"color": (80,  160, 255), "hex": "#50A0FF"},
    "Non-Pollution":       {"color": (80,  200,  80), "hex": "#50C850"},
}


# ─── Encode helper ───────────────────────────────────────────────────────────
def _enc(arr: np.ndarray, ext: str = ".png") -> str:
    import base64
    ok, buf = cv2.imencode(ext, arr)
    if not ok:
        return ""
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf).decode()


def _gray2bgr(g):
    return cv2.cvtColor((g > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)


# ─── Step 1: Preprocessing ───────────────────────────────────────────────────
def preprocess(img_bgr: np.ndarray, filter_type: str = "gaussian",
               ksize: int = 5):
    img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_AREA)
    from utils.segmentation import apply_gaussian, apply_median
    if filter_type == "median":
        filtered = apply_median(img_bgr, ksize)
    else:
        filtered = apply_gaussian(img_bgr, ksize)
    return img_bgr, filtered


# ─── Step 2: Enhancement ─────────────────────────────────────────────────────
def enhance(filtered: np.ndarray, method: str = "clahe"):
    from utils.segmentation import apply_histogram_eq, apply_clahe, apply_homomorphic
    if method == "histogram":
        return apply_histogram_eq(filtered)
    elif method == "homomorphic":
        return apply_homomorphic(filtered)
    else:
        return apply_clahe(filtered)


# ─── Step 3: Segmentation ────────────────────────────────────────────────────
def segment(enhanced: np.ndarray, method: str = "otsu"):
    from utils.segmentation import (segment_otsu, segment_adaptive,
                                     segment_canny, segment_color_range)
    if method == "adaptive":
        return segment_adaptive(enhanced)
    elif method == "canny":
        return segment_canny(enhanced)
    elif method == "color":
        return segment_color_range(enhanced)
    else:
        return segment_otsu(enhanced)


# ─── Step 4: Morphological Refinement ───────────────────────────────────────
def morph_refine(mask: np.ndarray, se_shape: str = "ellipse", se_size: int = 9):
    from utils.segmentation import apply_morphology
    return apply_morphology(mask, se_shape, se_size)


# ─── Step 5: Feature Extraction ──────────────────────────────────────────────
def extract_features(mask: np.ndarray, img_bgr: np.ndarray) -> list:
    """Extract features for every detected region. Returns list of region dicts."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    h_img, w_img = mask.shape[:2]
    total_area = h_img * w_img

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 100:  # skip tiny noise regions
            continue
        perimeter = float(cv2.arcLength(cnt, True))
        circularity = (4 * np.pi * area / (perimeter ** 2 + 1e-6))
        compactness = (perimeter ** 2) / (area + 1e-6)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / (h + 1e-6)
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / (hull_area + 1e-6)
        extent = area / (w * h + 1e-6)
        pct_image = 100.0 * area / total_area

        # Color stats within region
        region_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        cv2.drawContours(region_mask, [cnt], -1, 255, cv2.FILLED)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_sat  = float(hsv[:, :, 1][region_mask == 255].mean()) if region_mask.any() else 0
        mean_val  = float(hsv[:, :, 2][region_mask == 255].mean()) if region_mask.any() else 0
        mean_hue  = float(hsv[:, :, 0][region_mask == 255].mean()) if region_mask.any() else 0

        # Classify region
        label = classify_region(area, circularity, solidity, compactness,
                                  mean_sat, mean_val, pct_image, mean_hue)

        regions.append({
            "contour":      cnt,
            "area":         round(area, 1),
            "perimeter":    round(perimeter, 1),
            "circularity":  round(float(np.clip(circularity, 0, 1)), 4),
            "compactness":  round(compactness, 2),
            "aspect_ratio": round(float(np.clip(aspect_ratio, 0, 20)), 3),
            "solidity":     round(float(np.clip(solidity, 0, 1)), 4),
            "extent":       round(float(np.clip(extent, 0, 1)), 4),
            "pct_image":    round(pct_image, 2),
            "mean_hue":     round(mean_hue, 1),
            "mean_sat":     round(mean_sat, 1),
            "mean_val":     round(mean_val, 1),
            "bbox":         (x, y, w, h),
            "label":        label,
        })

    # Sort by area descending
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions


# ─── Step 6: Rule-Based Classification ──────────────────────────────────────
def classify_region(area, circularity, solidity, compactness,
                     mean_sat, mean_val, pct_image, mean_hue=0) -> str:
    """
    Rule-based pollution classifier — v2 (fixed thresholds).

    Priority order (most specific → least specific):
    1. Surface Reflection : very bright + near-white (specular glare)
    2. Foam Pollution      : bright white/grey, irregular, bubbly texture
    3. Oil Slick           : dark region, spread, low-sat OR iridescent sheen
    4. Debris/Sediment     : brownish-yellow mid-brightness
    5. Non-Pollution       : everything else (clean water, vegetation etc.)
    """
    # 1. Surface Reflection — pure specular highlights
    if mean_val > 220 and mean_sat < 25:
        return "Surface Reflection"

    # 2. Foam Pollution — white/grey, bright, irregular blobs
    if mean_val > 155 and mean_sat < 75:
        return "Foam Pollution"

    # 3. Oil Slick — very dark AND covers meaningful area
    if mean_val < 85 and pct_image > 0.3:
        return "Oil Slick"

    # 3b. Oil iridescent sheen — moderate brightness, some colour tint
    if 60 < mean_val < 140 and 20 < mean_sat < 130 and pct_image > 0.2:
        return "Oil Slick"

    # 4. Debris / Sediment — brownish hues (hue 10–35), mid brightness
    if 10 <= mean_hue <= 35 and mean_sat > 40 and 80 < mean_val < 180:
        return "Debris / Sediment"

    # 5. Default
    return "Non-Pollution"


# ─── Step 7: Visualisation ───────────────────────────────────────────────────
def build_visualisation(img_bgr: np.ndarray, regions: list) -> np.ndarray:
    """
    Draw filled semi-transparent regions + contour outlines + safe labels.
    Fixes:
      - No UTF-8 characters in putText (px^2 instead of px²)
      - Labels clipped to image bounds so they never overflow
      - Filled colour overlay per label for clear visual separation
    """
    vis = img_bgr.copy()
    overlay = vis.copy()

    # BGR colours for each class
    color_map = {
        "Foam Pollution":     (80,  80,  255),   # red-ish
        "Oil Slick":          (0,   140, 255),   # orange
        "Surface Reflection": (255, 180, 50),    # blue-yellow
        "Debris / Sediment":  (30,  180, 180),   # teal
        "Non-Pollution":      (60,  180, 60),    # green
    }

    h_img, w_img = vis.shape[:2]

    for r in regions:
        cnt   = r["contour"]
        label = r["label"]
        color = color_map.get(label, (180, 180, 180))

        # Semi-transparent filled region
        cv2.drawContours(overlay, [cnt], -1, color, cv2.FILLED)
        # Bold contour outline
        cv2.drawContours(vis, [cnt], -1, color, 2)

    # Blend overlay
    cv2.addWeighted(overlay, 0.28, vis, 0.72, 0, vis)

    # Draw labels on top (after blending)
    for r in regions:
        cnt   = r["contour"]
        label = r["label"]
        color = color_map.get(label, (180, 180, 180))
        x, y, w, h = r["bbox"]

        # Safe area text — ASCII only (cv2 cannot render UTF-8 superscripts)
        area_txt = f"{int(r['area'])} px2"

        # Label background — keep inside image
        txt_w = max(len(label) * 7 + 6, 60)
        lbl_y0 = max(0, y - 20)
        lbl_y1 = max(20, y)
        lbl_x1 = min(x + txt_w, w_img)

        cv2.rectangle(vis, (x, lbl_y0), (lbl_x1, lbl_y1), color, -1)
        cv2.putText(vis, label, (x + 2, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Area text below bbox — keep inside image
        area_y = min(y + h + 13, h_img - 3)
        cv2.putText(vis, area_txt, (x, area_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    return vis


# ─── Step 8: Robustness Variants ────────────────────────────────────────────
def apply_robustness_variant(img_bgr: np.ndarray, variant: str) -> np.ndarray:
    """Apply a distortion variant for robustness testing."""
    if variant == "gaussian_noise":
        noise = np.zeros_like(img_bgr, dtype=np.int16)
        cv2.randn(noise, 0, 35)
        return np.clip(img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif variant == "salt_pepper":
        out = img_bgr.copy()
        total = out.size // 3
        coords = [np.random.randint(0, i, total // 20) for i in img_bgr.shape[:2]]
        out[coords[0], coords[1]] = 255
        coords2 = [np.random.randint(0, i, total // 20) for i in img_bgr.shape[:2]]
        out[coords2[0], coords2[1]] = 0
        return out
    elif variant == "rotation":
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 25, 1.0)
        return cv2.warpAffine(img_bgr, M, (w, h))
    elif variant == "scaling":
        h, w = img_bgr.shape[:2]
        scaled = cv2.resize(img_bgr, (int(w * 0.6), int(h * 0.6)))
        return cv2.resize(scaled, (w, h))
    elif variant == "brightness":
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.7, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif variant == "compression":
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 10])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img_bgr.copy()


# ─── Main Pipeline ───────────────────────────────────────────────────────────
def run_pipeline(img_bgr: np.ndarray,
                 filter_type: str = "gaussian",
                 enhance_method: str = "clahe",
                 seg_method: str = "otsu",
                 se_shape: str = "ellipse",
                 se_size: int = 9,
                 location: str = "") -> dict:
    t0 = time.time()

    # Steps
    raw, filtered   = preprocess(img_bgr, filter_type)
    enhanced        = enhance(filtered, enhance_method)
    seg_mask        = segment(enhanced, seg_method)
    morph           = morph_refine(seg_mask, se_shape, se_size)
    final_mask      = morph["final"]
    regions         = extract_features(final_mask, raw)
    vis             = build_visualisation(raw, regions)

    process_time = round(time.time() - t0, 3)

    # Summary stats
    labels = [r["label"] for r in regions]
    label_counts = {}
    for lb in labels:
        label_counts[lb] = label_counts.get(lb, 0) + 1

    pollution_regions = [r for r in regions
                         if r["label"] in ("Foam Pollution", "Oil Slick")]
    total_pixels = raw.shape[0] * raw.shape[1]
    pollution_area_pct = round(
        100.0 * sum(r["area"] for r in pollution_regions) / total_pixels, 2)

    dominant = labels[0] if labels else "Non-Pollution"
    severity = ("Critical" if pollution_area_pct > 20
                else "Moderate" if pollution_area_pct > 5
                else "Low" if pollution_area_pct > 0.5
                else "Clean")

    images = {
        "raw":       _enc(raw),
        "filtered":  _enc(filtered),
        "enhanced":  _enc(enhanced),
        "segmented": _enc(_gray2bgr(seg_mask)),
        "eroded":    _enc(_gray2bgr(morph["eroded"])),
        "dilated":   _enc(_gray2bgr(morph["dilated"])),
        "opened":    _enc(_gray2bgr(morph["opened"])),
        "closed":    _enc(_gray2bgr(morph["closed"])),
        "final_mask":_enc(_gray2bgr(final_mask)),
        "visualised":_enc(vis),
    }

    return {
        "regions":            [_serialise_region(r) for r in regions],
        "label_counts":       label_counts,
        "dominant_label":     dominant,
        "pollution_area_pct": pollution_area_pct,
        "severity":           severity,
        "total_regions":      len(regions),
        "process_time":       process_time,
        "images":             images,
        "params": {
            "filter": filter_type, "enhance": enhance_method,
            "segment": seg_method, "se_shape": se_shape, "se_size": se_size,
        },
        "location": location,
    }


def _serialise_region(r: dict) -> dict:
    """Remove non-serialisable contour array."""
    return {k: v for k, v in r.items() if k != "contour"}


# ─── Algorithm Combination Study ─────────────────────────────────────────────
def run_combination_study(img_bgr: np.ndarray) -> list:
    """Test multiple pipeline combinations and return comparison results."""
    combos = [
        {"filter": "gaussian",  "enhance": "histogram",    "segment": "otsu",
         "label": "Gaussian + HistEq + Otsu"},
        {"filter": "gaussian",  "enhance": "clahe",        "segment": "otsu",
         "label": "Gaussian + CLAHE + Otsu"},
        {"filter": "median",    "enhance": "clahe",        "segment": "otsu",
         "label": "Median + CLAHE + Otsu"},
        {"filter": "median",    "enhance": "homomorphic",  "segment": "otsu",
         "label": "Median + Homomorphic + Otsu"},
        {"filter": "gaussian",  "enhance": "clahe",        "segment": "adaptive",
         "label": "Gaussian + CLAHE + Adaptive"},
        {"filter": "gaussian",  "enhance": "clahe",        "segment": "canny",
         "label": "Gaussian + CLAHE + Canny"},
        {"filter": "median",    "enhance": "clahe",        "segment": "color",
         "label": "Median + CLAHE + Color Range"},
    ]
    results = []
    for c in combos:
        r = run_pipeline(img_bgr, c["filter"], c["enhance"], c["segment"])
        results.append({
            "label":         c["label"],
            "filter":        c["filter"],
            "enhance":       c["enhance"],
            "segment":       c["segment"],
            "regions_found": r["total_regions"],
            "pollution_pct": r["pollution_area_pct"],
            "severity":      r["severity"],
            "process_time":  r["process_time"],
            "preview":       r["images"]["visualised"],
        })
    return results


# ─── Robustness Study ────────────────────────────────────────────────────────
def run_robustness_study(img_bgr: np.ndarray) -> list:
    variants = [
        ("original",       "Original"),
        ("gaussian_noise", "Gaussian Noise"),
        ("salt_pepper",    "Salt & Pepper Noise"),
        ("rotation",       "Rotation 25°"),
        ("scaling",        "Scaling 60%→100%"),
        ("brightness",     "Brightness ×1.7"),
        ("compression",    "JPEG Q=10 (Compressed)"),
    ]
    results = []
    baseline = run_pipeline(img_bgr)
    base_pct = baseline["pollution_area_pct"]

    for key, name in variants:
        if key == "original":
            r = baseline
        else:
            distorted = apply_robustness_variant(img_bgr, key)
            r = run_pipeline(distorted)

        drift = round(abs(r["pollution_area_pct"] - base_pct), 2)
        stability = ("Stable" if drift < 2 else
                     "Moderate" if drift < 8 else "Unstable")
        results.append({
            "variant":       key,
            "label":         name,
            "regions_found": r["total_regions"],
            "pollution_pct": r["pollution_area_pct"],
            "severity":      r["severity"],
            "process_time":  r["process_time"],
            "drift":         drift,
            "stability":     stability,
            "preview":       r["images"]["visualised"],
        })
    return results


# ─── Comparative Analysis (Assignments 1→2→3) ───────────────────────────────
def run_comparative_analysis(img_bgr: np.ndarray) -> list:
    """Simulate basic / enhanced / full-integrated pipeline comparison."""
    img = cv2.resize(img_bgr, (640, 480))

    # Basic (Assignment-1): just Gaussian + Otsu, no enhancement
    t0 = time.time()
    filt = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
    _, mask_basic = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    regions_basic = extract_features(mask_basic, img)
    vis_basic = build_visualisation(img, regions_basic)
    t_basic = round(time.time() - t0, 3)

    # Enhanced (Assignment-1+2): CLAHE + Otsu + basic morph
    t0 = time.time()
    enhanced = apply_clahe_local(img)
    gray2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, mask_enh = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_enh = cv2.morphologyEx(mask_enh, cv2.MORPH_OPEN, k)
    regions_enh = extract_features(mask_enh, img)
    vis_enh = build_visualisation(img, regions_enh)
    t_enh = round(time.time() - t0, 3)

    # Full integrated (Assignment-3)
    t0 = time.time()
    full = run_pipeline(img_bgr)
    t_full = round(time.time() - t0, 3)
    regions_full_count = full["total_regions"]

    return [
        {
            "system": "Basic (Assignment-1)",
            "techniques": "Gaussian Filter + Otsu Threshold",
            "regions": len(regions_basic),
            "pollution_pct": round(100.0 * sum(r["area"] for r in regions_basic if r["label"] != "Non-Pollution") / (640 * 480), 2),
            "noise_robust": "Low",
            "process_time": t_basic,
            "preview": _enc(vis_basic),
        },
        {
            "system": "Enhanced (Assignment-1+2)",
            "techniques": "Gaussian + CLAHE + Otsu + Opening",
            "regions": len(regions_enh),
            "pollution_pct": round(100.0 * sum(r["area"] for r in regions_enh if r["label"] != "Non-Pollution") / (640 * 480), 2),
            "noise_robust": "Medium",
            "process_time": t_enh,
            "preview": _enc(vis_enh),
        },
        {
            "system": "Full Integrated (Assignment-3)",
            "techniques": "Median/Gaussian + CLAHE/Homomorphic + Multi-seg + Full Morph + Rule Classifier",
            "regions": regions_full_count,
            "pollution_pct": full["pollution_area_pct"],
            "noise_robust": "High",
            "process_time": t_full,
            "preview": full["images"]["visualised"],
        },
    ]


def apply_clahe_local(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
