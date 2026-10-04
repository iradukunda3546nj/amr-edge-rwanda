"""
Digit-crop normalisation shared by training and inference.

The browser (src/web/js/core/digit-preprocess.js) and firmware implement the
same algorithm. Any change here must be mirrored there; the parity test in
tools/test/preprocess.test.mjs checks the JS port against fixtures written by
src/model/train.py.

Algorithm (input: grayscale uint8 crop of one counter wheel, any size):
  1. Separable area-average resize to 24x32 (W x H), round half up.
  2. Polarity: Otsu threshold on the central 60% x 70% window. If more than
     half of those pixels are above the threshold, the background is bright,
     so the crop is inverted. Output is always a bright digit on a dark drum,
     which is the polarity the model is trained on.
  3. Robust contrast stretch: map the 2nd..98th percentile to 0..255.
  4. Model input int8 = pixel - 128 (input scale 1/255, zero point -128).
"""

from __future__ import annotations

import numpy as np

OUT_W, OUT_H = 24, 32
N_PIX = OUT_W * OUT_H


def _area_taps(src: int, dst: int) -> list[tuple[int, list[float]]]:
    """For each destination index: first source index and normalised coverage weights."""
    scale = src / dst
    taps = []
    for i in range(dst):
        lo, hi = i * scale, (i + 1) * scale
        j0, j1 = int(np.floor(lo)), min(int(np.ceil(hi)), src)
        w = [min(hi, j + 1) - max(lo, j) for j in range(j0, j1)]
        total = 0.0
        for v in w:
            total += v
        taps.append((j0, [v / total for v in w]))
    return taps


def area_resize(img: np.ndarray, out_w: int = OUT_W, out_h: int = OUT_H) -> np.ndarray:
    """Separable area-average resize. Accumulates taps in the same order as the
    JS port (digit-preprocess.js), so both produce bit-identical results."""
    src = img.astype(np.float64)
    h, _ = src.shape
    tmp = np.empty((h, out_w))
    for i, (j0, ws) in enumerate(_area_taps(src.shape[1], out_w)):
        acc = np.zeros(h)
        for k, wk in enumerate(ws):
            acc = acc + src[:, j0 + k] * wk
        tmp[:, i] = acc
    out = np.empty((out_h, out_w))
    for i, (j0, ws) in enumerate(_area_taps(h, out_h)):
        acc = np.zeros(out_w)
        for k, wk in enumerate(ws):
            acc = acc + tmp[j0 + k, :] * wk
        out[i] = acc
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


def otsu_threshold(values: np.ndarray) -> int:
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sum_all = float(np.dot(np.arange(256), hist))
    w_b = sum_b = 0.0
    best_t, best_var = 0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def percentile_bounds(img: np.ndarray, lo_frac: float = 0.02, hi_frac: float = 0.98) -> tuple[int, int]:
    cum = np.cumsum(np.bincount(img.ravel(), minlength=256))
    n = img.size
    lo = int(np.searchsorted(cum, int(np.ceil(lo_frac * n))))
    hi = int(np.searchsorted(cum, int(np.ceil(hi_frac * n))))
    return lo, max(hi, lo + 1)


def normalize_digit(crop: np.ndarray) -> np.ndarray:
    """uint8 HxW grayscale crop -> uint8 32x24 normalised digit."""
    x = area_resize(crop)
    cy0, cy1 = int(OUT_H * 0.15), int(OUT_H * 0.85)
    cx0, cx1 = int(OUT_W * 0.20), int(OUT_W * 0.80)
    centre = x[cy0:cy1, cx0:cx1]
    t = otsu_threshold(centre)
    if np.count_nonzero(centre > t) * 2 > centre.size:
        x = 255 - x
    lo, hi = percentile_bounds(x)
    y = (x.astype(np.float64) - lo) * 255.0 / (hi - lo)
    return np.clip(np.floor(y + 0.5), 0, 255).astype(np.uint8)
