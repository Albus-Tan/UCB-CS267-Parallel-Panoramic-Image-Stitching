import os
import subprocess
import argparse
import tempfile
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def downsample_images(src_dir, dst_dir, scale):
    """Downsample all images in src_dir by the given scale and save to dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img = cv2.imread(os.path.join(src_dir, fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            img2 = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(dst_dir, fname), img2)

def run_experiment(input_dir, threads):
    """Run the panorama stitcher under OMP_NUM_THREADS and return time in seconds."""
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    result = subprocess.run(
        ['./pano.sh', 'run', 'openmp', '--dir', input_dir],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True
    )
    time_s = None
    for line in result.stdout.splitlines():
        if 'Total Execution Time (OpenMP):' in line:
            parts = line.split(':')[-1].strip().split()
            time_ms = float(parts[0])
            time_s = time_ms / 1000.0
            break
    if time_s is None:
        raise RuntimeError(f"Time not found in output:\n{result.stdout}")
    return time_s

def main():
    parser = argparse.ArgumentParser(description="Benchmark strong & weak scaling for panorama.")
    parser.add_argument('input_dir', help="Directory of source images")
    parser.add_argument('--output-csv', default='benchmark_results.csv', help="CSV file to save results")
    args = parser.parse_args()

    threads_list = [1, 2, 4, 8, 16, 32, 64]
    total_runs = len(threads_list) * 2
    run_count = 0
    records = []
    max_thread = max(threads_list)

    print(f"Starting benchmark: {total_runs} runs total (strong + weak)")

    # Strong scaling: fixed workload
    print("\n[Strong Scaling]")
    strong_dir = tempfile.mkdtemp(prefix='strong_')
    shutil.copytree(args.input_dir, strong_dir, dirs_exist_ok=True)
    for t in threads_list:
        run_count += 1
        print(f"  Run {run_count}/{total_runs} - strong: threads={t}", end=' ... ', flush=True)
        t_sec = run_experiment(strong_dir, t)
        print(f"{t_sec:.3f}s")
        records.append({'mode': 'strong', 'threads': t, 'scale': 1.0, 'time_sec': t_sec})
    shutil.rmtree(strong_dir)

    # Weak scaling: keep per-thread pixel count constant by downsampling
    print("\n[Weak Scaling]")
    for t in threads_list:
        run_count += 1
        scale = np.sqrt(t / max_thread)
        print(f"  Run {run_count}/{total_runs} - weak: threads={t}, scale=sqrt({t}/{max_thread})={scale:.3f}", end=' ... ', flush=True)
        weak_dir = tempfile.mkdtemp(prefix='weak_')
        downsample_images(args.input_dir, weak_dir, scale)
        t_sec = run_experiment(weak_dir, t)
        print(f"{t_sec:.3f}s")
        records.append({'mode': 'weak', 'threads': t, 'scale': scale, 'time_sec': t_sec})
        shutil.rmtree(weak_dir)

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)

    # Plot strong scaling
    plt.figure()
    strong_df = df[df['mode'] == 'strong']
    plt.loglog(strong_df['threads'], strong_df['time_sec'], 'o-', label='Actual')
    t1 = strong_df['time_sec'].iloc[0]
    ideal = t1 / np.array(strong_df['threads'])
    plt.loglog(strong_df['threads'], ideal, 'k--', label='Ideal (slope=-1)')
    plt.xlabel('Threads')
    plt.ylabel('Time (s)')
    plt.title('Strong Scaling')
    plt.legend()
    plt.grid(True)
    plt.savefig('strong_scaling.jpg')
    plt.close()

    # Plot weak scaling
    plt.figure()
    weak_df = df[df['mode'] == 'weak']
    plt.loglog(weak_df['threads'], weak_df['time_sec'], 'o-', label='Actual')

    # ideal flat line: use time at smallest thread count
    weak_sorted = weak_df.sort_values('threads')
    t_base = weak_sorted.iloc[0]['time_sec']
    plt.hlines(t_base,
               xmin=weak_sorted['threads'].min(),
               xmax=weak_sorted['threads'].max(),
               colors='k', linestyles='dashed',
               label='Ideal (flat)')
    plt.xlabel('Threads')
    plt.ylabel('Time (s)')
    plt.title('Weak Scaling')
    plt.legend()
    plt.grid(True)
    plt.savefig('weak_scaling.jpg')
    plt.close()

# python benchmark_panorama.py ./images/oilseed --output-csv results.csv
if __name__ == "__main__":
    main()
