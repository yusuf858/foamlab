"""
FoamLab — Segmentation utilities  (v2 — fixed)
Multi-method segmentation: Otsu, Canny, Adaptive, Color-Range
Key fix: segmentation now preserves BOTH bright foam AND dark oil regions
         without incorrectly inverting the mask.
"""
import numpy as np
import cv2


# ─── Preprocessing filters ───────────────────────────────────────────────────

def apply_gaussian(img_bgr: np.ndarray, ksize: int = 5) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img_bgr, (k, k), 0)


def apply_median(img_bgr: np.ndarray, ksize: int = 5) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(img_bgr, k)


# ─── Enhancement ─────────────────────────────────────────────────────────────

def apply_histogram_eq(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_clahe(img_bgr: np.ndarray, clip: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_homomorphic(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = np.where(gray == 0, 1e-5, gray)
    log_img = np.log(gray)
    rows, cols = log_img.shape
    fft_shift = np.fft.fftshift(np.fft.fft2(log_img))
    D0 = 30
    u = np.arange(-rows // 2, rows // 2)
    v = np.arange(-cols // 2, cols // 2)
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    H = 1.0 * (1 - np.exp(-D**2 / (2 * D0**2))) + 0.5
    result = np.exp(np.real(np.fft.ifft2(np.fft.ifftshift(fft_shift * H))))
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# ─── Segmentation methods ─────────────────────────────────────────────────────

def segment_otsu(img_bgr: np.ndarray) -> np.ndarray:
    """
    Multi-band Otsu: separately threshold for bright foam AND dark oil,
    then combine — avoids the single-inversion mistake.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Bright regions (foam / white debris)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dark regions (oil slick, dark contamination)
    # Invert the image so dark → bright, then Otsu
    inv = cv2.bitwise_not(blur)
    _, dark_inv = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove very large background blob from dark mask (keep only meaningful dark areas)
    # by requiring dark regions to be < 40% of image
    dark_pct = dark_inv.sum() / 255 / dark_inv.size
    dark_mask = dark_inv if dark_pct < 0.40 else np.zeros_like(dark_inv)

    combined = cv2.bitwise_or(bright, dark_mask)

    # Clean small noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k, iterations=1)
    return combined


def segment_adaptive(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 6)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def segment_canny(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Auto Canny thresholds based on image median
    med = float(np.median(blur))
    lo  = max(0,   int(0.66 * med))
    hi  = min(255, int(1.33 * med))
    edges = cv2.Canny(blur, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(edges, k, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    return mask


def segment_color_range(img_bgr: np.ndarray) -> np.ndarray:
    """
    Pollution-specific color segmentation:
      Foam    — bright white/grey (high V, low S)
      Oil     — dark iridescent (low V, low-mid S)
      Debris  — brownish/yellowish contaminants
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Foam: very bright, near-white
    foam = cv2.inRange(hsv,
                       np.array([0,   0, 170], np.uint8),
                       np.array([180, 55, 255], np.uint8))

    # Oil slick: dark, low saturation (dark grey/black patches)
    oil_dark = cv2.inRange(hsv,
                            np.array([0,  0,  0], np.uint8),
                            np.array([180, 80, 90], np.uint8))

    # Oil iridescent: medium brightness, some colour (rainbow sheen)
    oil_sheen = cv2.inRange(hsv,
                             np.array([0,  30, 60], np.uint8),
                             np.array([180, 140, 160], np.uint8))

    # Debris (brownish/yellowish)
    debris = cv2.inRange(hsv,
                          np.array([10, 50, 80], np.uint8),
                          np.array([35, 200, 200], np.uint8))

    combined = cv2.bitwise_or(foam, oil_dark)
    combined = cv2.bitwise_or(combined, oil_sheen)
    combined = cv2.bitwise_or(combined, debris)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k, iterations=1)
    return combined


# ─── Morphological refinement ────────────────────────────────────────────────

def apply_morphology(mask: np.ndarray, se_shape: str = "ellipse",
                     se_size: int = 9) -> dict:
    se_size = max(3, se_size | 1)
    shape_map = {"rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
    shape = shape_map.get(se_shape, cv2.MORPH_ELLIPSE)
    kernel = cv2.getStructuringElement(shape, (se_size, se_size))

    eroded  = cv2.erode(mask, kernel, iterations=1)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    opened  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    closed  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Final refined: open to remove noise, close to fill holes
    refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)

    return {
        "eroded": eroded, "dilated": dilated,
        "opened": opened, "closed": closed, "final": refined,
    }


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out
