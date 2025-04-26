import cv2
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim

def compute_metrics(img1, img2, reproj_threshold=3.0):
    # --- Crop to common region ---
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)
    # center‐crop both images
    y1, x1 = (h1 - h) // 2, (w1 - w) // 2
    y2, x2 = (h2 - h) // 2, (w2 - w) // 2
    img1_c = img1[y1:y1+h, x1:x1+w]
    img2_c = img2[y2:y2+h, x2:x2+w]

    # PSNR
    psnr_val = cv2.PSNR(img1_c, img2_c)
    
    # SSIM (handle color images)
    ssim_val, _ = ssim(img1_c, img2_c, full=True, channel_axis=2)

    # Feature‐based inlier ratio via ORB + RANSAC
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1_c, None)
    kp2, des2 = orb.detectAndCompute(img2_c, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        raise RuntimeError("Not enough matches for homography")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_threshold)
    mask = mask.ravel().astype(bool)
    inlier_ratio = mask.sum() / len(mask)

    # Average reprojection error (inliers only)
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1,3).T
    pts2_proj_h = (H @ pts1_h)
    pts2_proj_h /= pts2_proj_h[2]
    reproj = np.linalg.norm(pts2_proj_h[:2].T - pts2, axis=1)
    reproj_err = reproj[mask].mean()

    # Seam smoothness via gradient on the diff image
    diff = cv2.absdiff(img1_c, img2_c)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    seam_smoothness = np.mean(np.sqrt(gx*gx + gy*gy))

    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'Inlier Ratio': inlier_ratio,
        'Reprojection Error': reproj_err,
        'Seam Smoothness': seam_smoothness
    }

def evaluate_quality(metrics):
    # Reference guidelines
    guidelines = {
        'PSNR':             (30, 40),    # (good threshold, excellent threshold) [dB]
        'SSIM':             (0.90, 0.95),
        'Inlier Ratio':     (0.50, 0.70), # (acceptable, good)
        'Reprojection Error': (1.0, 0.5), # (good pixel error <=1px, excellent <=0.5px)
        'Seam Smoothness':  (20, 10)     # (acceptable, good) low is better
    }
    quality_scores = {'good': 0, 'acceptable': 0, 'poor': 0}
    
    for name, value in metrics.items():
        low, high = guidelines[name]
        if name == 'Seam Smoothness':
            if value <= high:
                category = 'good'
            elif value <= low:
                category = 'acceptable'
            else:
                category = 'poor'
        elif name in ('Reprojection Error',):
            if value <= high:
                category = 'good'
            elif value <= low:
                category = 'acceptable'
            else:
                category = 'poor'
        else:
            # higher is better
            if value >= high:
                category = 'good'
            elif value >= low:
                category = 'acceptable'
            else:
                category = 'poor'
        quality_scores[category] += 1
        print(f"{name}: {value:.4f} ({category}, ref: {low} - {high})")

    # Overall quality
    if quality_scores['poor'] > 0:
        overall = 'Poor'
    elif quality_scores['acceptable'] > 0:
        overall = 'Acceptable'
    else:
        overall = 'Good'
    print(f"\nOverall stitching quality: {overall}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', help="Baseline panorama")
    parser.add_argument('test',     help="Test panorama")
    parser.add_argument('--threshold', type=float, default=3.0,
                        help="RANSAC reproj threshold")
    args = parser.parse_args()

    img1 = cv2.imread(args.baseline)
    img2 = cv2.imread(args.test)
    if img1 is None or img2 is None:
        raise RuntimeError("Could not load images")

    metrics = compute_metrics(img1, img2, reproj_threshold=args.threshold)
    evaluate_quality(metrics)

if __name__ == "__main__":
    main()

