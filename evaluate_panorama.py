import cv2
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim

def compute_metrics(img1, img2, reproj_threshold=3.0):
    # --- Compute homography and inlier mask via ORB + RANSAC ---
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        raise RuntimeError("Not enough matches for homography")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, inlier_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_threshold)
    if H is None:
        raise RuntimeError("Homography estimation failed")
    inlier_mask = inlier_mask.ravel().astype(bool)
    inlier_ratio = inlier_mask.sum() / len(inlier_mask)

    # Average reprojection error
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1,3).T
    pts2_proj_h = (H @ pts1_h)
    pts2_proj_h /= pts2_proj_h[2]
    proj_pts = pts2_proj_h[:2].T
    reproj_err = np.linalg.norm(proj_pts[inlier_mask] - pts2[inlier_mask], axis=1).mean()

    # --- Warp baseline into test frame and build overlap mask ---
    h2, w2 = img2.shape[:2]
    warped = cv2.warpPerspective(img1, H, (w2, h2),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0,0,0))
    overlap_mask = np.any(warped != 0, axis=2)
    if not overlap_mask.any():
        raise RuntimeError("No overlap region found")

    # --- PSNR over overlap ---
    diff = (warped.astype(np.float32) - img2.astype(np.float32))
    mse = np.mean(diff[overlap_mask] ** 2)
    psnr_val = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')

    # --- SSIM over overlap ---
    ssim_val = ssim(warped, img2, channel_axis=2, mask=overlap_mask)

    # --- Seam smoothness: gradient magnitude along seam boundary ---
    gray_diff = cv2.cvtColor(cv2.absdiff(warped, img2), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray_diff, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray_diff, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(gx*gx + gy*gy)

    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(overlap_mask.astype(np.uint8), kernel)
    seam_mask = dilated.astype(bool) & (~overlap_mask)
    seam_smoothness = grad_mag[seam_mask].mean() if seam_mask.any() else 0.0

    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'Inlier Ratio': inlier_ratio,
        'Reprojection Error': reproj_err,
        'Seam Smoothness': seam_smoothness
    }

def evaluate_quality(metrics):
    # Define thresholds: (acceptable threshold, good threshold)
    guidelines = {
        'PSNR':               (25, 35),   # dB
        'SSIM':               (0.80, 0.90),
        'Inlier Ratio':       (0.50, 0.70),
        'Reprojection Error': (3.0, 1.0), # px
        'Seam Smoothness':    (30, 10)    # gradient units
    }

    # Print level definitions and ranges
    print("Quality Levels:")
    print("  good        : meets or exceeds the 'good' threshold")
    print("  acceptable  : meets or exceeds the 'acceptable' threshold, but below 'good'")
    print("  poor        : below the 'acceptable' threshold\n")

    print("Metric Ranges:")
    for name, (acc, good) in guidelines.items():
        if name in ('Reprojection Error', 'Seam Smoothness'):
            print(f"  {name:17s}: good ≤ {good}, acceptable ≤ {acc}, else poor")
        else:
            print(f"  {name:17s}: good ≥ {good}, acceptable ≥ {acc}, else poor")
    print()

    # Evaluate each metric
    counts = {'good':0, 'acceptable':0, 'poor':0}
    for name, val in metrics.items():
        acc, good = guidelines[name]
        if name in ('Reprojection Error', 'Seam Smoothness'):
            is_good = val <= good
            is_acc  = val <= acc
        else:
            is_good = val >= good
            is_acc  = val >= acc

        if is_good:
            category = 'good'
        elif is_acc:
            category = 'acceptable'
        else:
            category = 'poor'
        counts[category] += 1
        print(f"{name:17s}: {val:.4f} [{category}]")

    # Overall quality
    if counts['poor'] > 0:
        overall = 'Poor'
    elif counts['acceptable'] > 0:
        overall = 'Acceptable'
    else:
        overall = 'Good'
    print(f"\nOverall stitching quality: {overall}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', help="Baseline panorama image")
    parser.add_argument('test',     help="Test panorama image")
    parser.add_argument('--threshold', type=float, default=3.0,
                        help="RANSAC reproj threshold in pixels")
    args = parser.parse_args()

    img1 = cv2.imread(args.baseline)
    img2 = cv2.imread(args.test)
    if img1 is None or img2 is None:
        raise RuntimeError("Failed to load images")

    metrics = compute_metrics(img1, img2, reproj_threshold=args.threshold)
    evaluate_quality(metrics)

if __name__ == "__main__":
    main()
