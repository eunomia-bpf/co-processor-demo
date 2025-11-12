#!/usr/bin/env python3
"""
Create run scripts for all benchmarks that don't have them
"""

import os
from pathlib import Path

def create_run_script(benchmark_dir, command):
    """Create a run script in benchmark directory"""
    run_script = benchmark_dir / 'run'

    with open(run_script, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(command + '\n')

    os.chmod(run_script, 0o755)
    print(f"Created: {run_script}")

def main():
    root = Path.cwd()

    # Polybench - most just need to run the .exe file
    polybench_dirs = [
        '2DCONV', '2MM', '3DCONV', '3MM', 'ATAX', 'BICG', 'CORR',
        'COVAR', 'FDTD-2D', 'GEMM', 'GESUMMV', 'GRAMSCHM', 'MVT',
        'SYR2K', 'SYRK'
    ]

    for version in ['UVM_benchmarks', 'non_UVM_benchmarks']:
        print(f"\nCreating run scripts for {version}/polybench/...")

        for bench in polybench_dirs:
            bench_dir = root / version / 'polybench' / bench

            if not bench_dir.exists():
                continue

            # Find the executable
            exe_files = list(bench_dir.glob('*.exe'))
            if not exe_files:
                # Try to find any executable
                for f in bench_dir.iterdir():
                    if f.is_file() and os.access(f, os.X_OK) and not f.name.startswith('.'):
                        exe_files = [f]
                        break

            if exe_files:
                exe_name = exe_files[0].name
                create_run_script(bench_dir, f'./{exe_name}')

    # Rodinia benchmarks - more complex, need data files
    rodinia_configs = {
        'backprop': './backprop 65536',
        'dwt2d': './dwt2d ../../../data/dwt2d/192.bmp -d 192x192 -f -5 -l 3',
        'gaussian': './gaussian -s 1024',
        'hotspot': './hotspot 512 2 2 ../../../data/hotspot/temp_512 ../../../data/hotspot/power_512 output.out',
        'hotspot3D': './hotspot3D 512 8 100 ../../../data/hotspot3D/power_512x8 ../../../data/hotspot3D/temp_512x8 output.out',
        'nw': './needle 2048 10',
        'particlefilter': './particlefilter -x 128 -y 128 -z 10 -np 1000',
        'pathfinder': './pathfinder 100000 100 20',
        'srad': './srad 100 0.5 502 458',
        'streamcluster': './streamcluster 10 20 256 65536 65536 1000 none output.txt 1',
    }

    for version in ['UVM_benchmarks', 'non_UVM_benchmarks']:
        print(f"\nCreating run scripts for {version}/rodinia/...")

        for bench, cmd in rodinia_configs.items():
            bench_dir = root / version / 'rodinia' / bench

            if bench_dir.exists():
                create_run_script(bench_dir, cmd)

    # Create a list of known broken benchmarks
    broken_file = root / 'KNOWN_BROKEN.txt'
    with open(broken_file, 'w') as f:
        f.write("# Benchmarks known to have issues\n\n")
        f.write("## Segmentation Faults (need deeper fixes):\n")
        f.write("- BN (ordergraph): Crashes immediately, needs debugging\n")
        f.write("- KNN: Crashes with current problem size\n")
        f.write("- bfs: CUDA driver version mismatch issue\n\n")
        f.write("## Missing Data Files:\n")
        f.write("- SVM: Needs training data in data/SVM/\n\n")
        f.write("## Successfully Running:\n")
        f.write("- CNN\n")
        f.write("- kmeans\n")
        f.write("- logistic-regression\n")
        f.write("- polybench: 2MM, 3MM, ATAX, BICG, GEMM, GESUMMV, MVT, SYR2K, SYRK\n")

    print(f"\nCreated: {broken_file}")
    print("\nDone!")

if __name__ == '__main__':
    main()
