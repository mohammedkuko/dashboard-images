
import os
from typing import Tuple, Optional, List
import cv2
import numpy as np

TARGET_W, TARGET_H = 675, 995

def log(msg: str): print(msg, flush=True)

def _find_bbox(img: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    h, w = img.shape[:2]
    left_cut = int(w * 0.33)
    roi = img[:, left_cut:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 45, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dil = cv2.dilate(closed, kernel, iterations=1)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_AREA_FRAC = 0.20
    best = None; best_score = -1.0
    for c in cnts:
        x,y,ww,hh = cv2.boundingRect(c)
        x_full = x + left_cut
        y_full = y
        area = ww * hh
        area_frac = area / float(h*w)
        if area_frac < MIN_AREA_FRAC:
            continue
        aspect = ww / float(hh + 1e-6)
        if not (0.45 <= aspect <= 0.95):
            continue
        cx = x_full + ww / 2.0; cy = y_full + hh / 2.0
        rightness = cx / w; downness = cy / h
        rect_ratio = min(ww, hh) / max(ww, hh)
        squareness_penalty = (1 - rect_ratio) * 0.12
        score = area_frac * (rightness**1.75) * (downness**1.25) - squareness_penalty
        if score > best_score:
            best_score = score; best = (x_full, y_full, ww, hh)

    if best is None:
        target_ratio = TARGET_W / float(TARGET_H)
        for scale in [0.9, 0.88, 0.86, 0.84, 0.82, 0.8, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68]:
            H = int(h * scale); W = int(H * target_ratio)
            x1 = int(w * 0.98); y1 = int(h * 0.98)
            x0 = x1 - W; y0 = y1 - H
            if x0 > left_cut + int(0.01*w) and y0 >= 0 and W>0 and H>0:
                best = (x0, y0, W, H); break

    if best is None: return None

    px = int(0.01 * w); py = int(0.01 * h)
    x,y,ww,hh = best
    x0 = max(0, x - px); y0 = max(0, y - py); x1 = min(w, x + ww + px); y1 = min(h, y + hh + py)
    return (x0, y0, x1-x0, y1-y0)

def _trim_white_border(img: np.ndarray) -> np.ndarray:
    """Remove uniform white/near-white border around the poster.
       Uses HSV: low saturation + high value considered 'white'.
       Scans from each edge until at least ~1% of pixels in that row/col are non-white.
    """
    if img.size == 0: return img
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # thresholds: tuneable; S <= 40, V >= 230 considered white/near-white
    white = (hsv[:,:,1] <= 40) & (hsv[:,:,2] >= 230)
    nonwhite = ~white

    # If almost everything is white (unlikely), bail out
    if nonwhite.sum() < 0.001 * h * w:
        return img

    row_counts = nonwhite.sum(axis=1)  # per-row non-white pixels
    col_counts = nonwhite.sum(axis=0)  # per-col non-white pixels

    row_thresh = max(1, int(0.01 * w))  # at least 1% of columns must be non-white
    col_thresh = max(1, int(0.01 * h))  # at least 1% of rows must be non-white

    # scan from each side
    top = 0
    while top < h and row_counts[top] < row_thresh:
        top += 1
    bottom = h - 1
    while bottom > top and row_counts[bottom] < row_thresh:
        bottom -= 1
    left = 0
    while left < w and col_counts[left] < col_thresh:
        left += 1
    right = w - 1
    while right > left and col_counts[right] < col_thresh:
        right -= 1

    # small safety inset (avoid shaving content): add 1px margin back
    top = max(0, top - 1)
    left = max(0, left - 1)
    bottom = min(h - 1, bottom + 1)
    right = min(w - 1, right + 1)

    if right <= left or bottom <= top:
        return img
    return img[top:bottom+1, left:right+1]

def crop_and_resize(img: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    x,y,ww,hh = bbox
    crop = img[y:y+hh, x:x+ww]
    crop = _trim_white_border(crop)
    return cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

def process_file(in_path: str, out_path: str) -> Optional[Tuple[int,int,int,int]]:
    img = cv2.imread(in_path)
    if img is None: return None
    bbox = _find_bbox(img)
    if bbox is None: return None
    poster = crop_and_resize(img, bbox)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, poster)  # PNG by extension
    return bbox

def batch_crop(input_folder: str, output_folder: str, extensions=(".png",".jpg",".jpeg",".webp",".jfif")) -> List[Tuple[str, str, Optional[Tuple[int,int,int,int]]]]:
    files = []
    for root, _, fs in os.walk(input_folder):
        for f in fs:
            if f.lower().endswith(extensions):
                files.append(os.path.join(root, f))
    files.sort()
    total = len(files)
    os.makedirs(output_folder, exist_ok=True)
    results = []
    for i, in_fp in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(in_fp))[0] + ".png"
        out_fp = os.path.join(output_folder, base)
        # log(f"[{i}/{total}] Starting: {in_fp}")
        bbox = process_file(in_fp, out_fp)
        if bbox is None:
            log(f"[{i}/{total}] SKIPPED (no poster detected): {in_fp}")
        else:
            x,y,w,h = bbox
            # log(f"[{i}/{total}] Done: {in_fp} -> bbox=({x},{y},{w},{h}), saved {out_fp} ({TARGET_W}x{TARGET_H})")
        results.append((in_fp, out_fp, bbox))
    ok = sum(1 for _,_,b in results if b is not None)
    log(f"Processed {total} files, cropped {ok} posters, skipped {total-ok}. Output -> {output_folder}")
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Crop bottom-right posters, trim white borders, and resize to 675x995.")
    ap.add_argument("input_folder", help="Folder containing the screenshots")
    ap.add_argument("output_folder", help="Folder to write cropped posters (PNG)")
    args = ap.parse_args()
    batch_crop(args.input_folder, args.output_folder)
