"""
horse_detector.py

Profesionalni pipeline za detekciju konja:
- Trening SVM klasifikatora (HOG features) koristeći:
    HorsesData/pos  (pozitivni primeri + groundtruth)
    HorsesData/neg  (negativni primeri)
- Sliding-window + image pyramid detekcija na test slikama
- Spajanje detekcija (NMS)
- Evaluacija po Jaccard index (IoU) koristeći groundtruth fajlove
Usage:
    python horse_detector.py --data_dir ./HorsesData --mode train
    python horse_detector.py --data_dir ./HorsesData --mode eval
    python horse_detector.py --data_dir ./HorsesData --mode detect --image path/to/image.jpg
"""

import os
import glob
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
from tqdm import tqdm
import argparse
import math
import matplotlib.pyplot as plt

# -------------------------
#  CONFIGURATION / HYPERPARAMS
# -------------------------
MODEL_PATH = "horse_svm_hog.joblib"
PATCH_SIZE = (128, 128)        # fixed size to which we resize training patches
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "feature_vector": True
}
NEG_WINDOWS_PER_IMAGE = 10     # number of negative windows sampled per negative image
SCALES = [1.0, 0.9, 0.8, 0.7, 0.6]  # image pyramid scales used during detection
WINDOW_STRIDE = 32             # sliding window stride (in pixels)
DECISION_THRESHOLD = 0.0       # SVM decision function threshold: > threshold -> positive
NMS_IOU_THRESH = 0.3           # IoU threshold for Non-Max Suppression
IOU_POSITIVE_THRESH = 0.5      # IoU threshold considered a correct detection during evaluation

# -------------------------
#  UTIL: groundtruth reading
# -------------------------
def read_groundtruth_for_image(gt_folder, image_filename):
    """
    Reads groundtruth file for image. Groundtruth file naming assumed:
    <image_basename>__entires.groundtruth (as in task specification).
    Returns bbox as tuple (x1, y1, x2, y2) or None if file missing.
    """
    base = os.path.splitext(os.path.basename(image_filename))[0]
    # file name per spec:
    gt_name = f"{base}__entires.groundtruth"
    gt_path = os.path.join(gt_folder, gt_name)
    if not os.path.exists(gt_path):
        return None
    with open(gt_path, "r") as f:
        line = f.readline().strip()
        if not line:
            return None
        parts = line.split()
        if len(parts) < 4:
            return None
        x1, y1, x2, y2 = map(float, parts[:4])
        return (int(x1), int(y1), int(x2), int(y2))

# -------------------------
#  UTIL: HOG extraction
# -------------------------
def compute_hog(img_gray, hog_params=HOG_PARAMS):
    # expects grayscale image (numpy array). returns 1D feature vector
    return hog(img_gray,
               orientations=hog_params["orientations"],
               pixels_per_cell=hog_params["pixels_per_cell"],
               cells_per_block=hog_params["cells_per_block"],
               block_norm=hog_params["block_norm"],
               feature_vector=hog_params["feature_vector"])

# -------------------------
#  UTIL: Non-Maximum Suppression
# -------------------------
def iou(boxA, boxB):
    # boxes as (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0.0
    return interArea / union

def non_max_suppression(boxes, scores, iou_threshold=NMS_IOU_THRESH):
    """
    boxes: list of (x1,y1,x2,y2)
    scores: corresponding confidence scores
    returns: list of boxes after NMS
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    idxs = np.argsort(scores)[::-1]  # descending scores
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]
        rem = []
        for j in rest:
            if iou(boxes[i], boxes[j]) <= iou_threshold:
                rem.append(j)
        idxs = np.array(rem)
    return boxes[keep].tolist(), scores[keep].tolist()

# -------------------------
#  TRAIN: prepare dataset
# -------------------------
def prepare_training_data(data_dir):
    """
    - positive patches: from HorsesData/pos, read each image's groundtruth and crop that bbox.
      Resize to PATCH_SIZE.
    - negative patches: from HorsesData/neg, sample random windows (no horse).
    Returns X (hog vectors), y labels (1 for pos, 0 for neg)
    """
    pos_dir = os.path.join(data_dir, "pos")
    neg_dir = os.path.join(data_dir, "neg")
    # groundtruth folder: assume groundtruth files are in same directory as pos images (or a folder).
    # We'll look for groundtruth files in pos_dir as well.
    X = []
    y = []

    # Positives
    pos_images = sorted(glob.glob(os.path.join(pos_dir, "*.*")))
    for img_path in tqdm(pos_images, desc="Collecting positives"):
        gt = read_groundtruth_for_image(pos_dir, img_path)
        if gt is None:
            # skip if no groundtruth found
            continue
        x1, y1, x2, y2 = gt
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # clip coordinates to image
        h, w = gray.shape
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w - 1, x2), min(h - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        patch = gray[y1c:y2c, x1c:x2c]
        patch_resized = cv2.resize(patch, PATCH_SIZE)
        feat = compute_hog(patch_resized)
        X.append(feat)
        y.append(1)

    # Negatives: sample random windows from negative images
    neg_images = sorted(glob.glob(os.path.join(neg_dir, "*.*")))
    rng = np.random.RandomState(42)
    for img_path in tqdm(neg_images, desc="Collecting negatives"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        for k in range(NEG_WINDOWS_PER_IMAGE):
            if h < 16 or w < 16:
                continue
            # sample random top-left corner so window fits
            win_w, win_h = PATCH_SIZE
            if w <= win_w or h <= win_h:
                # if negative image smaller than patch size, resize full image
                patch = cv2.resize(gray, PATCH_SIZE)
            else:
                x = rng.randint(0, w - win_w)
                y0 = rng.randint(0, h - win_h)
                patch = gray[y0:y0 + win_h, x:x + win_w]
            feat = compute_hog(patch)
            X.append(feat)
            y.append(0)

    X = np.array(X)
    y = np.array(y)
    return X, y

# -------------------------
#  TRAIN: train and save model
# -------------------------
def train_model(data_dir, model_path=MODEL_PATH):
    print("Preparing training data...")
    X, y = prepare_training_data(data_dir)
    print("Feature shape:", X.shape, "Labels:", np.bincount(y))
    # simple train/test split for monitoring
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # pipeline: scaler + LinearSVC
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", LinearSVC(max_iter=20000, dual=False))
    ])
    print("Training SVM...")
    pipeline.fit(X_train, y_train)
    print("Train acc:", accuracy_score(y_train, pipeline.predict(X_train)))
    print("Holdout acc:", accuracy_score(y_hold, pipeline.predict(X_hold)))
    # persist model
    dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

# -------------------------
#  DETECTION: sliding window + pyramid
# -------------------------
def sliding_window(img_gray, window_size, stride):
    """
    Yields (x, y, window) top-left corner coordinates and window content.
    """
    win_w, win_h = window_size
    h, w = img_gray.shape
    for y in range(0, h - win_h + 1, stride):
        for x in range(0, w - win_w + 1, stride):
            yield (x, y, img_gray[y:y + win_h, x:x + win_w])

def image_pyramid(img, scales=SCALES):
    """
    yields (scale, resized_image)
    """
    h0, w0 = img.shape[:2]
    for s in scales:
        new_w = int(w0 * s)
        new_h = int(h0 * s)
        if new_w < PATCH_SIZE[0] or new_h < PATCH_SIZE[1]:
            continue
        resized = cv2.resize(img, (new_w, new_h))
        yield s, resized

def detect_in_image(image_path, pipeline, visualize=False):
    """
    Runs detection on a single image using sliding window + pyramid.
    Returns list of final boxes (x1,y1,x2,y2) in original image coordinates and scores.
    """
    img = cv2.imread(image_path)
    if img is None:
        return [], []
    orig_h, orig_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_boxes = []
    detected_scores = []

    for scale, im_scaled in image_pyramid(gray):
        scaled_h, scaled_w = im_scaled.shape
        # sliding window on scaled image
        for (x, y, window) in sliding_window(im_scaled, PATCH_SIZE, WINDOW_STRIDE):
            if window.shape[0] != PATCH_SIZE[1] or window.shape[1] != PATCH_SIZE[0]:
                continue
            feat = compute_hog(window).reshape(1, -1)
            score = pipeline.decision_function(feat)[0]  # linear SVM decision value
            if score > DECISION_THRESHOLD:
                # map coords back to original image
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + PATCH_SIZE[0] - 1) / scale)
                y2 = int((y + PATCH_SIZE[1] - 1) / scale)
                # clip
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w - 1, x2), min(orig_h - 1, y2)
                detected_boxes.append((x1, y1, x2, y2))
                detected_scores.append(float(score))

    # apply NMS
    boxes_nms, scores_nms = non_max_suppression(detected_boxes, detected_scores, iou_threshold=NMS_IOU_THRESH)
    if visualize:
        vis = img.copy()
        for (bx, by, bx2, by2) in boxes_nms:
            cv2.rectangle(vis, (bx, by), (bx2, by2), (0, 255, 0), 2)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    return boxes_nms, scores_nms

# -------------------------
#  EVALUATION
# -------------------------
def evaluate_on_testset(data_dir, model_path=MODEL_PATH):
    test_dir = os.path.join(data_dir, "test")
    pipeline = load(model_path)
    results = []
    iou_list = []
    detect_count = 0
    test_images = sorted(glob.glob(os.path.join(test_dir, "*.*")))
    for img_path in tqdm(test_images, desc="Evaluating test images"):
        gt = read_groundtruth_for_image(test_dir, img_path)
        pred_boxes, pred_scores = detect_in_image(img_path, pipeline, visualize=False)
        # choose best predicted box by score, else empty
        if len(pred_boxes) == 0:
            iou_val = 0.0
        else:
            # compute IoU of each predicted box with GT (single GT per image)
            if gt is None:
                iou_val = 0.0
            else:
                ious = [iou(box, gt) for box in pred_boxes]
                iou_val = max(ious)  # best overlap
        iou_list.append(iou_val)
        if iou_val >= IOU_POSITIVE_THRESH:
            detect_count += 1
        results.append((img_path, gt, pred_boxes, pred_scores, iou_val))
    mean_iou = np.mean(iou_list) if len(iou_list) > 0 else 0.0
    detection_rate = detect_count / len(test_images) if len(test_images) > 0 else 0.0
    print(f"Test images: {len(test_images)}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Detection rate (IoU >= {IOU_POSITIVE_THRESH}): {detection_rate:.3f}")
    return results

# -------------------------
#  CLI / main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to HorsesData folder")
    parser.add_argument("--mode", type=str, choices=["train", "detect", "eval"], default="train")
    parser.add_argument("--image", type=str, default=None, help="Single image path for detect mode")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to model file")
    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.data_dir, model_path=args.model)
    elif args.mode == "detect":
        if args.image is None:
            print("Provide --image for detect mode")
            return
        pipeline = load(args.model)
        boxes, scores = detect_in_image(args.image, pipeline, visualize=True)
        print("Detections:", boxes, scores)
    elif args.mode == "eval":
        results = evaluate_on_testset(args.data_dir, model_path=args.model)
        # optional: save results
        out_file = "eval_results.txt"
        with open(out_file, "w") as f:
            for r in results:
                f.write(f"{r[0]}\tGT:{r[1]}\tPredCount:{len(r[2])}\tIoU:{r[4]:.4f}\n")
        print("Detailed results saved to", out_file)


if __name__ == "__main__":
    main()
