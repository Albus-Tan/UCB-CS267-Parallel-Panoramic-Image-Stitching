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
            img2 = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
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
    for line in result.stdout.splitlines():
        if 'Total Execution Time (OpenMP):' in line:
            ms = float(line.split(':')[-1].strip().split()[0])
            return ms / 1000.0
    raise RuntimeError(f"Execution time not found in output:\n{result.stdout}")

def benchmark_dataset(dataset_dir, dataset_name, threads_list, max_thread, run_count, total_runs, records):
    """Runs strong and weak scaling for one dataset, updating records list and run_count."""
    # Strong scaling
    strong_dir = tempfile.mkdtemp(prefix=f'strong_{dataset_name}_')
    shutil.copytree(dataset_dir, strong_dir, dirs_exist_ok=True)
    for t in threads_list:
        run_count += 1
        print(f"  Run {run_count}/{total_runs} - {dataset_name} strong: threads={t}", end=' ... ', flush=True)
        t_sec = run_experiment(strong_dir, t)
        print(f"{t_sec:.3f}s")
        records.append({'dataset': dataset_name, 'mode': 'strong', 'threads': t,
                        'scale': 1.0, 'time_sec': t_sec})
    shutil.rmtree(strong_dir)

    # Weak scaling: downsample so per-thread work is constant
    for t in threads_list:
        run_count += 1
        scale = np.sqrt(t / max_thread)
        print(f"  Run {run_count}/{total_runs} - {dataset_name} weak: threads={t}, scale={scale:.3f}", end=' ... ', flush=True)
        weak_dir = tempfile.mkdtemp(prefix=f'weak_{dataset_name}_')
        downsample_images(dataset_dir, weak_dir, scale)
        t_sec = run_experiment(weak_dir, t)
        print(f"{t_sec:.3f}s")
        records.append({'dataset': dataset_name, 'mode': 'weak', 'threads': t,
                        'scale': scale, 'time_sec': t_sec})
        shutil.rmtree(weak_dir)

    return run_count

def main():
    parser = argparse.ArgumentParser(description="Benchmark strong & weak scaling for panoramas.")
    parser.add_argument('input_dir', help="Directory of source images or parent directory")
    parser.add_argument('--all', action='store_true',
                        help="If set, run on all subdirectories of input_dir")
    parser.add_argument('--output-csv', default='benchmark_results.csv',
                        help="CSV file to save results")
    args = parser.parse_args()

    # Determine datasets
    if args.all:
        dataset_dirs = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir)
                        if os.path.isdir(os.path.join(args.input_dir, d))]
    else:
        dataset_dirs = [args.input_dir]
    dataset_dirs = sorted(dataset_dirs)
    dataset_names = [os.path.basename(d) for d in dataset_dirs]

    threads_list = [1, 2, 4, 8, 16, 32, 64]
    max_thread = max(threads_list)
    total_runs = len(dataset_dirs) * len(threads_list) * 2
    run_count = 0
    records = []

    print(f"Starting benchmark on {len(dataset_dirs)} dataset(s): {dataset_names}")
    print(f"Total runs: {total_runs} (strong + weak for each)")

    # Run benchmarks
    for dataset_dir, dataset_name in zip(dataset_dirs, dataset_names):
        print(f"\n=== Dataset: {dataset_name} ===")
        run_count = benchmark_dataset(dataset_dir, dataset_name,
                                      threads_list, max_thread,
                                      run_count, total_runs, records)

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)

    # Plot strong scaling for all datasets
    plt.figure()
    for ds in dataset_names:
        ds_df = df[(df['dataset'] == ds) & (df['mode'] == 'strong')]
        plt.loglog(ds_df['threads'], ds_df['time_sec'], marker='o', label=ds)
    # ideal line
    t1 = df[(df['mode']=='strong') & (df['threads']==1)]['time_sec'].min()
    ideal = t1 / np.array(threads_list)
    plt.loglog(threads_list, ideal, 'k--', label='Ideal')
    plt.xlabel('Threads')
    plt.ylabel('Time (s)')
    plt.title('Strong Scaling')
    plt.legend()
    plt.grid(True)
    plt.savefig('strong_scaling.jpg')
    plt.close()

    # Plot weak scaling for all datasets
    plt.figure()
    for ds in dataset_names:
        ds_df = df[(df['dataset'] == ds) & (df['mode'] == 'weak')]
        plt.loglog(ds_df['threads'], ds_df['time_sec'], marker='o', label=ds)
    # ideal flat
    t_base = df[(df['mode']=='weak') & (df['threads']==1)]['time_sec'].min()
    plt.hlines(t_base, min(threads_list), max(threads_list),
               colors='k', linestyles='dashed', label='Ideal')
    plt.xlabel('Threads')
    plt.ylabel('Time (s)')
    plt.title('Weak Scaling')
    plt.legend()
    plt.grid(True)
    plt.savefig('weak_scaling.jpg')
    plt.close()

# python benchmark_panorama.py --all ./images --output-csv all_results.csv
# python benchmark_panorama.py ./images/oilseed --output-csv oilseed.csv
if __name__ == "__main__":
    main()
