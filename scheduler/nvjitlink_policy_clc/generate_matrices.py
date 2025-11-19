#!/usr/bin/env python3
"""
Generate test matrices for GEMM benchmarking
Creates different matrix patterns to test various workload characteristics
"""

import numpy as np
import os
import struct
import argparse

def write_matrix_binary(filename, matrix):
    """Write matrix to binary file in row-major float32 format"""
    with open(filename, 'wb') as f:
        # Write dimensions (M, N)
        f.write(struct.pack('ii', matrix.shape[0], matrix.shape[1]))
        # Write data
        matrix.astype(np.float32).tofile(f)

def generate_uniform(M, N, seed=42):
    """Generate uniform random matrix [0, 1)"""
    np.random.seed(seed)
    return np.random.rand(M, N).astype(np.float32)

def generate_sparse(M, N, sparsity=0.9, seed=42):
    """Generate sparse matrix with given sparsity ratio"""
    np.random.seed(seed)
    matrix = np.random.rand(M, N).astype(np.float32)
    mask = np.random.rand(M, N) < sparsity
    matrix[mask] = 0.0
    return matrix

def generate_diagonal_heavy(M, N, seed=42):
    """Generate matrix with heavy diagonal elements"""
    np.random.seed(seed)
    matrix = np.random.rand(M, N).astype(np.float32) * 0.1
    min_dim = min(M, N)
    for i in range(min_dim):
        matrix[i, i] = 10.0 + np.random.rand()
    return matrix

def generate_block_structured(M, N, block_size=64, seed=42):
    """Generate block-structured matrix (simulates tiled computation)"""
    np.random.seed(seed)
    matrix = np.zeros((M, N), dtype=np.float32)
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            block_m = min(block_size, M - i)
            block_n = min(block_size, N - j)
            matrix[i:i+block_m, j:j+block_n] = np.random.rand(block_m, block_n)
    return matrix

def generate_imbalanced(M, N, seed=42):
    """Generate imbalanced matrix (first rows heavy, last rows light)"""
    np.random.seed(seed)
    matrix = np.random.rand(M, N).astype(np.float32)
    # Make first 20% of rows heavier (larger values)
    heavy_rows = M // 5
    matrix[:heavy_rows, :] *= 10.0
    # Make last 20% of rows lighter (smaller values)
    light_rows = M // 5
    matrix[-light_rows:, :] *= 0.1
    return matrix

def generate_identity(M, N):
    """Generate identity-like matrix"""
    matrix = np.zeros((M, N), dtype=np.float32)
    min_dim = min(M, N)
    for i in range(min_dim):
        matrix[i, i] = 1.0
    return matrix

def generate_all_matrices(output_dir, sizes):
    """Generate all matrix types for given sizes"""
    os.makedirs(output_dir, exist_ok=True)

    patterns = {
        'uniform': generate_uniform,
        'sparse_50': lambda M, N, s: generate_sparse(M, N, 0.5, s),
        'sparse_90': lambda M, N, s: generate_sparse(M, N, 0.9, s),
        'diagonal': generate_diagonal_heavy,
        'block': generate_block_structured,
        'imbalanced': generate_imbalanced,
        'identity': lambda M, N, s: generate_identity(M, N),
    }

    manifest = []

    for M, N, K in sizes:
        print(f"\nGenerating matrices for size M={M}, N={N}, K={K}")
        size_dir = os.path.join(output_dir, f"{M}x{N}x{K}")
        os.makedirs(size_dir, exist_ok=True)

        for pattern_name, pattern_func in patterns.items():
            print(f"  Creating {pattern_name} pattern...")

            # Generate A (M x K)
            if pattern_name == 'identity':
                A = pattern_func(M, K, 42)
            else:
                A = pattern_func(M, K, 42)

            # Generate B (K x N)
            if pattern_name == 'identity':
                B = pattern_func(K, N, 43)
            else:
                B = pattern_func(K, N, 43)

            # Generate C (M x N) - initialize to zeros
            C = np.zeros((M, N), dtype=np.float32)

            # Write matrices
            A_file = os.path.join(size_dir, f"A_{pattern_name}.bin")
            B_file = os.path.join(size_dir, f"B_{pattern_name}.bin")
            C_file = os.path.join(size_dir, f"C_{pattern_name}.bin")

            write_matrix_binary(A_file, A)
            write_matrix_binary(B_file, B)
            write_matrix_binary(C_file, C)

            manifest.append({
                'size': f"{M}x{N}x{K}",
                'pattern': pattern_name,
                'A': A_file,
                'B': B_file,
                'C': C_file,
                'M': M, 'N': N, 'K': K
            })

    # Write manifest file
    manifest_file = os.path.join(output_dir, 'manifest.txt')
    with open(manifest_file, 'w') as f:
        f.write("# Matrix test manifest\n")
        f.write("# Format: size,pattern,A_file,B_file,C_file\n")
        for entry in manifest:
            f.write(f"{entry['size']},{entry['pattern']},{entry['A']},{entry['B']},{entry['C']}\n")

    print(f"\n✓ Generated {len(manifest)} matrix sets")
    print(f"✓ Manifest written to {manifest_file}")

    return manifest

def main():
    parser = argparse.ArgumentParser(description='Generate test matrices for GEMM benchmarking')
    parser.add_argument('--output-dir', '-o', default='./test_matrices',
                        help='Output directory for matrices (default: ./test_matrices)')
    parser.add_argument('--sizes', '-s', nargs='+',
                        default=['512x512x512', '1024x1024x1024', '256x256x256', '2048x2048x2048'],
                        help='Matrix sizes in format MxNxK (default: 512x512x512 1024x1024x1024 256x256x256 2048x2048x2048)')

    args = parser.parse_args()

    # Parse sizes
    sizes = []
    for size_str in args.sizes:
        M, N, K = map(int, size_str.split('x'))
        sizes.append((M, N, K))

    print("=" * 60)
    print("GEMM Matrix Generator")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Sizes: {sizes}")
    print("=" * 60)

    generate_all_matrices(args.output_dir, sizes)

    print("\nMatrix patterns generated:")
    print("  - uniform: Random uniform distribution [0, 1)")
    print("  - sparse_50: 50% sparsity")
    print("  - sparse_90: 90% sparsity")
    print("  - diagonal: Heavy diagonal elements")
    print("  - block: Block-structured (64x64 blocks)")
    print("  - imbalanced: First 20% rows heavy, last 20% light")
    print("  - identity: Identity-like matrix")

if __name__ == '__main__':
    main()
