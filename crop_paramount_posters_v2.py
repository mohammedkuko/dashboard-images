
import os
from pathlib import Path
from typing import Tuple, Optional, List
import cv2
import numpy as np

TARGET_W, TARGET_H = 675, 995

def log(msg: str):
    print(msg, flush=True)

def find_poster_bbox(img: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    h, w = img.shape[:2]
    left_cut = int(w * 0.33)
    roi = img[:, left_cut:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dil = cv2.dilate(closed, kernel, iterations=1)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1.0
    for c in cnts:
        x,y,ww,hh = cv2.boundingRect(c)
        x_full = x + left_cut
        y_full = y
        area = ww * hh
        if area < 0.05 * h * w:
            continue
        aspect = ww / float(hh + 1e-6)
        if not (0.5 <= aspect <= 0.85):
            continue
        cx = x_full + ww / 2.0
        cy = y_full + hh / 2.0
        rightness = cx / w
        downness = cy / h
        rect_ratio = min(ww, hh) / max(ww, hh)
        squareness_penalty = (1 - rect_ratio) * 0.15
        score = (area / (h*w)) * (rightness**1.75) * (downness**1.25) - squareness_penalty
        if score > best_score:
            best_score = score
            best = (x_full, y_full, ww, hh)
    if best is None:
        return None
    px = int(0.02 * w)
    py = int(0.02 * h)
    x,y,ww,hh = best
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(w, x + ww + px)
    y1 = min(h, y + hh + py)
    return (x0, y0, x1 - x0, y1 - y0)

def crop_and_resize(img: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    x,y,ww,hh = bbox
    crop = img[y:y+hh, x:x+ww]
    resized = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    return resized

def crop_poster_from_file(in_path: str, out_path: str) -> Optional[Tuple[int,int,int,int]]:
    img = cv2.imread(in_path)
    if img is None:
        return None
    bbox = find_poster_bbox(img)
    if bbox is None:
        return None
    result = crop_and_resize(img, bbox)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, result)  # always PNG by out_path extension
    return bbox

def batch_crop(input_folder: str, output_folder: str, extensions=(".png",".jpg",".jpeg",".webp",".jfif")) -> List[Tuple[str, str, Optional[Tuple[int,int,int,int]]]]:
    input_folder = str(input_folder)
    output_folder = str(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    results = []
    files = []
    for root, _, fs in os.walk(input_folder):
        for f in fs:
            if f.lower().endswith(extensions):
                files.append(os.path.join(root, f))
    files.sort()
    total = len(files)
    for idx, in_fp in enumerate(files, 1):
        base_name = os.path.splitext(os.path.basename(in_fp))[0] + ".png"
        out_fp = os.path.join(output_folder, base_name)
        # log(f"[{idx}/{total}] Starting: {in_fp}")
        bbox = crop_poster_from_file(in_fp, out_fp)
        if bbox is None:
            log(f"[{idx}/{total}] SKIPPED (no poster detected): {in_fp}")
        else:
            x,y,w,h = bbox
            # log(f"[{idx}/{total}] Done: {in_fp}  -> bbox=({x},{y},{w},{h}), saved {out_fp} ({TARGET_W}x{TARGET_H})")
        results.append((in_fp, out_fp, bbox))
    ok = sum(1 for _,_,b in results if b is not None)
    log(f"Processed {total} files, cropped {ok} posters, skipped {total-ok}. Output -> {output_folder}")
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Crop bottom-right posters from Paramount+ AB test screenshots.")
    ap.add_argument("input_folder", help="Folder containing the screenshots")
    ap.add_argument("output_folder", help="Folder to write cropped posters (PNG)")
    args = ap.parse_args()
    batch_crop(args.input_folder, args.output_folder)
