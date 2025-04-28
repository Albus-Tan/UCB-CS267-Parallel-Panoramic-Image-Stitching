#!/usr/bin/env python3
import os
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import csv

def run_serial(input_dir, out_prefix):
    """Runs the serial panorama stitcher, prints its output, and returns total time in seconds."""
    cmd = [
        './pano.sh', 'run', 'serial',
        '--dir', input_dir,
        '--out', f'{out_prefix}_serial_panorama.jpg'
    ]
    print(f"\n--- Running Serial on {input_dir} ---")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    # Print the stitcher’s own output to console
    print(result.stdout, end='')

    for line in result.stdout.splitlines():
        if 'Image Stitching:' in line:
            ms = float(line.split(':')[-1].strip().split()[0])
            return ms / 1000.0
    raise RuntimeError(f"Execution time not found in output:\n{result.stdout}")

def run_openmp(input_dir, out_prefix, threads=64):
    """Runs the OpenMP panorama stitcher, prints its output, and returns total time in seconds."""
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    cmd = [
        './pano.sh', 'run', 'openmp',
        '--dir', input_dir,
        '--out', f'{out_prefix}_openmp_panorama.jpg'
    ]
    print(f"\n--- Running OpenMP ({threads} threads) on {input_dir} ---")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True
    )
    # Print the stitcher’s own output to console
    print(result.stdout, end='')

    for line in result.stdout.splitlines():
        if 'Total Execution Time (OpenMP):' in line:
            ms = float(line.split(':')[-1].strip().split()[0])
            return ms / 1000.0
    raise RuntimeError(f"Execution time not found in output:\n{result.stdout}")

def main(root_dir):
    csv_file = 'benchmark_results.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'serial_time_s', 'openmp_time_s'])

    datasets = []
    serial_times = []
    openmp_times = []

    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue

        serial_time = run_serial(path, name)
        openmp_time = run_openmp(path, name)

        datasets.append(name)
        serial_times.append(serial_time)
        openmp_times.append(openmp_time)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, f"{serial_time:.6f}", f"{openmp_time:.6f}"])

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(len(datasets)*1.2, 4))
    ax.bar(x - width/2, serial_times, width, label='Serial')
    ax.bar(x + width/2, openmp_times, width, label='OpenMP (64 threads)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel('Time (s)')
    ax.set_title('Serial vs. OpenMP Panorama Stitching Performance')

    all_times = np.array(serial_times + openmp_times)
    positive = all_times[all_times > 0]
    if positive.max() / positive.min() > 10:
        ax.set_yscale('log')
        ax.set_ylabel('Time (s, log scale)')

    ax.legend()
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=200)
    plt.show()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Benchmark panorama stitching")
    p.add_argument('--root', default='./images/',
                   help="Root directory containing one folder per dataset")
    args = p.parse_args()
    main(args.root)
