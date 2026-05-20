import cv2
import numpy as np


def _odd(value: float, minimum: int, maximum: int) -> int:
    value = int(value)
    value = max(minimum, min(maximum, value))
    return value if value % 2 else value + 1


def _fill_contours(mask: np.ndarray, min_area: int) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filled, [contour], -1, 255, -1)
    return filled


def _road_surface_mask(img_bgr: np.ndarray, h: int) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    gray_surface = (
        (saturation < 115)
        & (value > 30)
        & (value < 245)
    ).astype(np.uint8) * 255

    roi = np.zeros_like(gray_surface)
    roi[int(h * 0.35):, :] = 255
    road_mask = cv2.bitwise_and(gray_surface, roi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if cv2.countNonZero(road_mask) < int(gray_surface.size * 0.08):
        road_mask[int(h * 0.35):, :] = 255
    return road_mask


def _threshold_from_roi(values: np.ndarray, default: int, percentile: int) -> int:
    if values.size == 0:
        return default
    return max(default, int(np.percentile(values, percentile)))


def _surface_damage_mask(
    gray: np.ndarray,
    road_mask: np.ndarray,
    edge_mask: np.ndarray,
    texture_mask: np.ndarray,
    total_pixels: int,
) -> np.ndarray:
    h, w = gray.shape[:2]
    blur_size = _odd(min(h, w) * 0.18, 21, 71)
    local_background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    contrast = cv2.absdiff(gray, local_background)

    road_values = contrast[road_mask > 0]
    contrast_threshold = _threshold_from_roi(road_values, 14, 72)
    _, contrast_mask = cv2.threshold(
        contrast, contrast_threshold, 255, cv2.THRESH_BINARY
    )

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    road_gradient_values = gradient[road_mask > 0]
    gradient_threshold = _threshold_from_roi(road_gradient_values, 22, 70)
    _, gradient_mask = cv2.threshold(
        gradient, gradient_threshold, 255, cv2.THRESH_BINARY
    )

    damaged_texture = cv2.bitwise_or(edge_mask, texture_mask)
    damaged_texture = cv2.bitwise_or(damaged_texture, gradient_mask)
    surface_mask = cv2.bitwise_and(contrast_mask, damaged_texture)
    surface_mask = cv2.bitwise_and(surface_mask, road_mask)

    kernel_size = _odd(min(h, w) * 0.055, 5, 21)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _fill_contours(surface_mask, max(20, int(total_pixels * 0.00025)))


def _is_damage_candidate(contour: np.ndarray, gray: np.ndarray, total_pixels: int) -> bool:
    area = cv2.contourArea(contour)
    min_area = max(60, int(total_pixels * 0.001))
    max_area = int(total_pixels * 0.60)
    if area < min_area or area > max_area:
        return False

    x, y, width, height = cv2.boundingRect(contour)
    if width < 8 or height < 8:
        return False

    aspect_ratio = width / float(height)
    if aspect_ratio < 0.20 or aspect_ratio > 5.50:
        return False

    extent = area / float(width * height)
    if extent < 0.06:
        return False

    pad = max(6, int(max(width, height) * 0.25))
    y0 = max(0, y - pad)
    y1 = min(gray.shape[0], y + height + pad)
    x0 = max(0, x - pad)
    x1 = min(gray.shape[1], x + width + pad)

    local = gray[y0:y1, x0:x1]
    local_mask = np.zeros(local.shape, dtype=np.uint8)
    shifted = contour - np.array([[[x0, y0]]], dtype=contour.dtype)
    cv2.drawContours(local_mask, [shifted], -1, 255, -1)

    inside = local[local_mask > 0]
    if inside.size == 0:
        return False

    local_median = float(np.median(local))
    candidate_dark_value = float(np.percentile(inside, 25))
    candidate_bright_value = float(np.percentile(inside, 75))
    dark_contrast = local_median - candidate_dark_value
    bright_contrast = candidate_bright_value - local_median

    return max(dark_contrast, bright_contrast) >= 8 or float(np.std(inside)) >= 18


def detect_damage(image_array: np.ndarray):
    """
    Detect pothole-like road damage with texture, edge, and dark-depression cues.

    Bright and dark road-surface patch cues are important for images where the
    road has water-filled potholes, patchy broken asphalt, or reflective pits.
    The older detector often erased these regions during cleanup and then
    reported zero potholes.
    """
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    adaptive_block = _odd(min(h, w) * 0.12, 31, 91)
    adaptive_dark = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=adaptive_block,
        C=6,
    )

    edges = cv2.Canny(blur, 20, 75)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_mask = cv2.dilate(edges, edge_kernel, iterations=1)

    density_kernel = np.ones((25, 25), np.float32) / (25 * 25)
    edge_density = cv2.filter2D(edges.astype(np.float32), -1, density_kernel)
    _, density_mask = cv2.threshold(edge_density, 7, 255, cv2.THRESH_BINARY)
    density_mask = density_mask.astype(np.uint8)

    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    lap_abs = np.abs(laplacian).astype(np.float32)
    lap_blur = cv2.GaussianBlur(lap_abs, (25, 25), 0)
    lap_norm = cv2.normalize(lap_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, texture_mask = cv2.threshold(lap_norm, 35, 255, cv2.THRESH_BINARY)

    dark_cutoff = int(np.percentile(blur, 22))
    dark_cutoff = max(45, min(125, dark_cutoff))
    global_dark = cv2.inRange(blur, 0, dark_cutoff)
    road_mask = _road_surface_mask(img_bgr, h)

    blackhat_size = _odd(min(h, w) * 0.14, 17, 61)
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (blackhat_size, blackhat_size)
    )
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, blackhat_kernel)
    blackhat_threshold = max(8, int(np.percentile(blackhat, 85)))
    _, blackhat_mask = cv2.threshold(
        blackhat, blackhat_threshold, 255, cv2.THRESH_BINARY
    )

    whitehat_size = _odd(min(h, w) * 0.12, 15, 51)
    whitehat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (whitehat_size, whitehat_size)
    )
    whitehat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, whitehat_kernel)
    road_whitehat_values = whitehat[road_mask > 0]
    whitehat_threshold = _threshold_from_roi(road_whitehat_values, 10, 84)
    _, whitehat_mask = cv2.threshold(
        whitehat, whitehat_threshold, 255, cv2.THRESH_BINARY
    )

    dark_with_edges = cv2.bitwise_and(global_dark, cv2.bitwise_or(edge_mask, blackhat_mask))
    dark_with_adaptive = cv2.bitwise_and(global_dark, adaptive_dark)
    depression_mask = cv2.bitwise_or(dark_with_edges, dark_with_adaptive)

    bridge_size = _odd(min(h, w) * 0.08, 7, 35)
    bridge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (bridge_size, bridge_size)
    )
    depression_mask = cv2.morphologyEx(
        depression_mask, cv2.MORPH_CLOSE, bridge_kernel, iterations=2
    )
    depression_mask = _fill_contours(depression_mask, max(20, int(total_pixels * 0.0004)))

    rough_surface_mask = cv2.bitwise_or(blackhat_mask, whitehat_mask)
    rough_surface_mask = cv2.bitwise_or(
        rough_surface_mask,
        cv2.bitwise_and(density_mask, texture_mask),
    )
    rough_surface_mask = cv2.bitwise_and(rough_surface_mask, road_mask)
    rough_surface_mask = cv2.morphologyEx(
        rough_surface_mask, cv2.MORPH_CLOSE, bridge_kernel, iterations=2
    )
    rough_surface_mask = _fill_contours(
        rough_surface_mask, max(15, int(total_pixels * 0.00025))
    )
    surface_damage_mask = _surface_damage_mask(
        gray, road_mask, edge_mask, texture_mask, total_pixels
    )

    vote = (
        (adaptive_dark > 0).astype(np.uint8)
        + (density_mask > 0).astype(np.uint8)
        + (texture_mask > 0).astype(np.uint8)
        + (global_dark > 0).astype(np.uint8)
    )
    fused = np.where(vote >= 2, 255, 0).astype(np.uint8)
    fused = cv2.bitwise_or(fused, depression_mask)
    fused = cv2.bitwise_or(fused, rough_surface_mask)
    fused = cv2.bitwise_or(fused, surface_damage_mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)
    cleaned = _fill_contours(cleaned, max(30, int(total_pixels * 0.0005)))

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    damage_contours = [
        contour
        for contour in contours
        if _is_damage_candidate(contour, gray, total_pixels)
    ]

    road_pixels = max(cv2.countNonZero(road_mask), 1)
    surface_damage_ratio = cv2.countNonZero(surface_damage_mask) / float(road_pixels)
    if not damage_contours and surface_damage_ratio >= 0.015:
        fallback_contours, _ = cv2.findContours(
            surface_damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        damage_contours = [
            contour
            for contour in fallback_contours
            if cv2.contourArea(contour) >= max(40, int(total_pixels * 0.00035))
        ]

    if not damage_contours and surface_damage_ratio >= 0.04:
        points = cv2.findNonZero(surface_damage_mask)
        if points is not None:
            x, y, bw, bh = cv2.boundingRect(points)
            damage_contours = [
                np.array(
                    [
                        [[x, y]],
                        [[x + bw, y]],
                        [[x + bw, y + bh]],
                        [[x, y + bh]],
                    ],
                    dtype=np.int32,
                )
            ]

    damage_contours.sort(key=cv2.contourArea, reverse=True)

    annotated = img_bgr.copy()
    severity_counts = {"small": 0, "medium": 0, "large": 0}
    total_damage_area = 0

    for contour in damage_contours:
        area = cv2.contourArea(contour)
        total_damage_area += area

        if area < total_pixels * 0.005:
            severity, color = "small", (0, 200, 80)
        elif area < total_pixels * 0.025:
            severity, color = "medium", (0, 140, 255)
        else:
            severity, color = "large", (0, 0, 220)

        severity_counts[severity] += 1

        x, y, bw, bh = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(
            annotated,
            severity[0].upper(),
            (x + 4, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    damage_area_pct = (total_damage_area / total_pixels) * 100

    stats = {
        "pothole_count": len(damage_contours),
        "damage_area_pct": round(damage_area_pct, 2),
        "severity": severity_counts,
    }

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return damage_contours, annotated_rgb, stats
