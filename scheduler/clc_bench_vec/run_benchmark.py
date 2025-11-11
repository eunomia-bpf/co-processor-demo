#!/usr/bin/env python3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# Test configurations
sizes = [
    (1024 * 256, "256K"),
    (1024 * 512, "512K"),
    (1024 * 1024, "1M"),
    (1024 * 1024 * 2, "2M"),
    (1024 * 1024 * 4, "4M"),
]
threads = 256

all_results = []

print("Running benchmarks...")
for size, label in sizes:
    print(f"Testing size: {label} elements")
    result = subprocess.run(
        ["./clc_benchmark_workloads", str(size), str(threads)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error running benchmark with size {label}")
        print(result.stderr)
        continue

    lines = result.stdout.strip().split('\n')

    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) >= 12:
            all_results.append({
                'Size': label,
                'Scenario': parts[0].split(':')[0],
                'FixedWork_ms': float(parts[2]),
                'FixedBlocks_ms': float(parts[4]),
                'CLC_ms': float(parts[6]),
            })

df = pd.DataFrame(all_results)

# Create single comparison figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('CLC vs Fixed Blocks vs Fixed Work - Execution Time Comparison', fontsize=16)

scenarios = df['Scenario'].unique()
size_order = ["256K", "512K", "1M", "2M", "4M"]

for idx, scenario in enumerate(scenarios):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    scenario_data = df[df['Scenario'] == scenario]
    scenario_data = scenario_data.set_index('Size').reindex(size_order).reset_index()

    x = range(len(scenario_data))
    width = 0.25

    ax.bar([i - width for i in x], scenario_data['FixedWork_ms'], width, label='Fixed Work', alpha=0.8)
    ax.bar(x, scenario_data['FixedBlocks_ms'], width, label='Fixed Blocks', alpha=0.8)
    ax.bar([i + width for i in x], scenario_data['CLC_ms'], width, label='CLC', alpha=0.8)

    ax.set_xlabel('Array Size')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title(scenario)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_data['Size'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('clc_benchmark_results.png', dpi=300, bbox_inches='tight')
print(f"\nSaved visualization to clc_benchmark_results.png")

# Calculate speedups
df['Speedup_vs_FixedWork'] = ((df['FixedWork_ms'] - df['CLC_ms']) / df['FixedWork_ms'] * 100)
df['Speedup_vs_FixedBlocks'] = ((df['FixedBlocks_ms'] - df['CLC_ms']) / df['FixedBlocks_ms'] * 100)

# Print detailed results in Markdown format
print("\n# CLC Benchmark Results - Detailed Speedup Report\n")

for size in size_order:
    size_data = df[df['Size'] == size]
    if len(size_data) == 0:
        continue

    print(f"\n## Size: {size} elements\n")
    print("| Workload | FixedWork (ms) | FixedBlock (ms) | CLC (ms) | vs FixWork | vs FixBlk |")
    print("|----------|----------------|-----------------|----------|------------|-----------|")

    for _, row in size_data.iterrows():
        speedup_fw = row['Speedup_vs_FixedWork']
        speedup_fb = row['Speedup_vs_FixedBlocks']

        # Format speedup with + or - sign
        speedup_fw_str = f"{speedup_fw:+.2f}%"
        speedup_fb_str = f"{speedup_fb:+.2f}%"

        print(f"| {row['Scenario']} | {row['FixedWork_ms']:.3f} | {row['FixedBlocks_ms']:.3f} | "
              f"{row['CLC_ms']:.3f} | {speedup_fw_str} | {speedup_fb_str} |")

# Print summary
print("\n---\n")
print("## Summary\n")

avg_speedup_fw = df['Speedup_vs_FixedWork'].mean()
avg_speedup_fb = df['Speedup_vs_FixedBlocks'].mean()

print("### Overall Average Speedup\n")
print(f"- **CLC vs Fixed Work**: {avg_speedup_fw:+.2f}%")
print(f"- **CLC vs Fixed Blocks**: {avg_speedup_fb:+.2f}%\n")

print("### Best Configurations\n")
print("| # | Workload | Size | vs FixWork | vs FixBlk | CLC Time (ms) |")
print("|---|----------|------|------------|-----------|---------------|")

top5 = df.nlargest(5, 'Speedup_vs_FixedBlocks')
for i, (idx, row) in enumerate(top5.iterrows(), 1):
    print(f"| {i} | {row['Scenario']} | {row['Size']} | {row['Speedup_vs_FixedWork']:+.2f}% | "
          f"{row['Speedup_vs_FixedBlocks']:+.2f}% | {row['CLC_ms']:.3f} |")

wins_fw = (df['Speedup_vs_FixedWork'] > 0).sum()
wins_fb = (df['Speedup_vs_FixedBlocks'] > 0).sum()
total = len(df)

print(f"\n### Win Rate\n")
print(f"- **vs Fixed Work**: {wins_fw}/{total} ({wins_fw/total*100:.1f}%)")
print(f"- **vs Fixed Blocks**: {wins_fb}/{total} ({wins_fb/total*100:.1f}%)\n")
