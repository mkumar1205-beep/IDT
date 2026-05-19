import cv2
import numpy as np


def detect_damage(image_array: np.ndarray):
    """
    Detect road damage using a multi-signal fusion approach:
      1. Texture anomaly  — potholes have rough/irregular texture vs smooth road
      2. Edge density     — damaged areas have high local edge concentration
      3. Dark region mask — deeper/shadowed potholes
      4. Adaptive thresh  — catches contrast boundaries

    This handles water-filled potholes that fool simple Canny detection.
    """
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Signal 1: CLAHE + Adaptive Threshold ─────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=35,
        C=6
    )

    # ── Signal 2: Canny edges ─────────────────────────────────────────────────
    edges = cv2.Canny(blur, 15, 60)

    # ── Signal 3: Edge DENSITY map (key for water-filled potholes) ───────────
    # Potholes have many edges packed together (cracks around the rim).
    # Smooth road has almost no edges. Convolve edge map with large kernel.
    density_kernel = np.ones((31, 31), np.float32) / (31 * 31)
    edge_density = cv2.filter2D(edges.astype(np.float32), -1, density_kernel)
    _, density_mask = cv2.threshold(edge_density, 12, 255, cv2.THRESH_BINARY)
    density_mask = density_mask.astype(np.uint8)

    # ── Signal 4: Texture anomaly via Laplacian variance ─────────────────────
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    lap_abs = np.abs(laplacian).astype(np.float32)
    lap_blur = cv2.GaussianBlur(lap_abs, (31, 31), 0)
    lap_norm = cv2.normalize(lap_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, texture_mask = cv2.threshold(lap_norm, 30, 255, cv2.THRESH_BINARY)

    # ── Signal 5: Dark region detection with Otsu auto-threshold ─────────────
    otsu_thresh, dark_mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    if otsu_thresh < 40 or otsu_thresh > 200:
        dark_mask = np.zeros_like(gray)

    # ── Fuse signals: require at least 2 signals to agree ────────────────────
    vote = (
        (thresh > 0).astype(np.uint8) +
        (density_mask > 0).astype(np.uint8) +
        (texture_mask > 0).astype(np.uint8) +
        (dark_mask > 0).astype(np.uint8)
    )
    fused = np.where(vote >= 2, 255, 0).astype(np.uint8)

    # ── Morphological cleanup ─────────────────────────────────────────────────
    kernel_close = np.ones((15, 15), np.uint8)
    kernel_open  = np.ones((7, 7),  np.uint8)
    closed  = cv2.morphologyEx(fused,  cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel_open)

    # ── Find contours ─────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    MIN_AREA = int(total_pixels * 0.003)
    MAX_AREA = int(total_pixels * 0.75)
    damage_contours = [
        c for c in contours
        if MIN_AREA < cv2.contourArea(c) < MAX_AREA
    ]

    # ── Draw bounding boxes ───────────────────────────────────────────────────
    annotated = img_bgr.copy()
    severity_counts = {"small": 0, "medium": 0, "large": 0}
    total_damage_area = 0

    for contour in damage_contours:
        area = cv2.contourArea(contour)
        total_damage_area += area

        if area < total_pixels * 0.005:
            severity, color = "small",  (0, 200, 80)
        elif area < total_pixels * 0.025:
            severity, color = "medium", (0, 140, 255)
        else:
            severity, color = "large",  (0, 0, 220)

        severity_counts[severity] += 1

        x, y, bw, bh = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(
            annotated, severity[0].upper(),
            (x + 4, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

    damage_area_pct = (total_damage_area / total_pixels) * 100

    stats = {
        "pothole_count":   len(damage_contours),
        "damage_area_pct": round(damage_area_pct, 2),
        "severity":        severity_counts,
    }

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return damage_contours, annotated_rgb, stats