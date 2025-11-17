"""
Performance benchmarking script for Gaussian Splatting rendering.

This script benchmarks the performance of the Gaussian Splatting layer
with different numbers of Gaussians to help identify performance bottlenecks
and optimal settings.

Usage:
    python benchmark_gaussian_rendering.py
"""

import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def benchmark_ply_io(n_gaussians_list):
    """Benchmark PLY file I/O performance."""
    print("\n" + "=" * 70)
    print("PLY File I/O Benchmark")
    print("=" * 70)

    try:
        from napari.layers.gaussians._gaussians_utils import (
            GaussianData,
            read_ply_gaussians,
            write_ply_gaussians,
        )
    except ImportError:
        print("ERROR: Could not import Gaussian utilities")
        return

    results = []

    for n in n_gaussians_list:
        print(f"\nTesting with {n:,} Gaussians...")

        # Create test data
        positions = np.random.rand(n, 3).astype(np.float32) * 100
        rotations = Rotation.random(n).as_quat().astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32) * 3 + 0.5
        opacities = np.random.rand(n).astype(np.float32)
        colors = np.random.rand(n, 3).astype(np.float32)

        data = GaussianData(
            positions=positions,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
        )

        # Benchmark write
        temp_path = Path(f'/tmp/benchmark_gaussians_{n}.ply')
        start = time.time()
        write_ply_gaussians(temp_path, data)
        write_time = time.time() - start

        # Benchmark read
        start = time.time()
        loaded_data = read_ply_gaussians(temp_path)
        read_time = time.time() - start

        # Clean up
        temp_path.unlink()

        # Calculate throughput
        write_throughput = n / write_time
        read_throughput = n / read_time

        print(f"  Write: {write_time:.3f}s ({write_throughput:,.0f} Gaussians/sec)")
        print(f"  Read:  {read_time:.3f}s ({read_throughput:,.0f} Gaussians/sec)")

        results.append({
            'n_gaussians': n,
            'write_time': write_time,
            'read_time': read_time,
            'write_throughput': write_throughput,
            'read_throughput': read_throughput,
        })

    return results


def benchmark_coordinate_transforms(n_gaussians_list):
    """Benchmark coordinate transformation performance."""
    print("\n" + "=" * 70)
    print("Coordinate Transformation Benchmark")
    print("=" * 70)

    results = []

    for n in n_gaussians_list:
        print(f"\nTesting with {n:,} Gaussians...")

        # Create test data
        positions = np.random.rand(n, 3).astype(np.float32)
        rotations = Rotation.random(n).as_quat().astype(np.float32)

        # Benchmark position reversal (ZYX <-> XYZ)
        start = time.time()
        positions_reversed = positions[:, ::-1]
        position_time = time.time() - start

        # Benchmark rotation transformation
        P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
        start = time.time()
        R_matrices = Rotation.from_quat(rotations).as_matrix()
        R_transformed = P @ R_matrices @ P.T
        rotations_transformed = Rotation.from_matrix(R_transformed).as_quat()
        rotation_time = time.time() - start

        position_throughput = n / position_time if position_time > 0 else float('inf')
        rotation_throughput = n / rotation_time

        print(f"  Position reversal: {position_time:.4f}s ({position_throughput:,.0f} Gaussians/sec)")
        print(f"  Rotation transform: {rotation_time:.4f}s ({rotation_throughput:,.0f} Gaussians/sec)")

        results.append({
            'n_gaussians': n,
            'position_time': position_time,
            'rotation_time': rotation_time,
            'position_throughput': position_throughput,
            'rotation_throughput': rotation_throughput,
        })

    return results


def benchmark_depth_sorting(n_gaussians_list):
    """Benchmark depth sorting performance."""
    print("\n" + "=" * 70)
    print("Depth Sorting Benchmark")
    print("=" * 70)

    try:
        from napari._vispy.visuals.gaussian_utils import compute_depth_order
    except ImportError:
        print("ERROR: Could not import gaussian_utils")
        return

    results = []
    view_matrix = np.eye(4, dtype=np.float32)

    for n in n_gaussians_list:
        print(f"\nTesting with {n:,} Gaussians...")

        positions = np.random.rand(n, 3).astype(np.float32) * 100

        # Benchmark depth sorting
        start = time.time()
        sort_indices = compute_depth_order(positions, view_matrix)
        sort_time = time.time() - start

        sort_throughput = n / sort_time

        print(f"  Depth sort: {sort_time:.4f}s ({sort_throughput:,.0f} Gaussians/sec)")

        results.append({
            'n_gaussians': n,
            'sort_time': sort_time,
            'sort_throughput': sort_throughput,
        })

    return results


def benchmark_rendering_utils(n_gaussians_list):
    """Benchmark rendering utility functions."""
    print("\n" + "=" * 70)
    print("Rendering Utilities Benchmark")
    print("=" * 70)

    try:
        from napari._vispy.visuals.gaussian_utils import (
            apply_opacity_to_colors,
            compute_gaussian_sizes,
            filter_by_opacity_threshold,
        )
    except ImportError:
        print("ERROR: Could not import gaussian_utils")
        return

    results = []

    for n in n_gaussians_list:
        print(f"\nTesting with {n:,} Gaussians...")

        positions = np.random.rand(n, 3).astype(np.float32)
        rotations = np.random.rand(n, 4).astype(np.float32)
        scales = np.random.rand(n, 3).astype(np.float32)
        opacities = np.random.rand(n).astype(np.float32)
        colors = np.random.rand(n, 3).astype(np.float32)

        # Benchmark size computation
        start = time.time()
        sizes = compute_gaussian_sizes(scales)
        size_time = time.time() - start

        # Benchmark color/opacity application
        start = time.time()
        rgba = apply_opacity_to_colors(colors, opacities)
        color_time = time.time() - start

        # Benchmark opacity filtering
        start = time.time()
        filtered = filter_by_opacity_threshold(
            positions, rotations, scales, opacities, colors
        )
        filter_time = time.time() - start

        print(f"  Size computation: {size_time:.4f}s ({n/size_time:,.0f} Gaussians/sec)")
        print(f"  Color application: {color_time:.4f}s ({n/color_time:,.0f} Gaussians/sec)")
        print(f"  Opacity filtering: {filter_time:.4f}s ({n/filter_time:,.0f} Gaussians/sec)")

        results.append({
            'n_gaussians': n,
            'size_time': size_time,
            'color_time': color_time,
            'filter_time': filter_time,
        })

    return results


def print_summary(all_results):
    """Print summary of all benchmarks."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\nKey Findings:")
    print("-" * 70)

    # Find bottlenecks
    if 'rotation_transform' in all_results:
        rotation_results = all_results['rotation_transform']
        slowest_rotation = min(rotation_results, key=lambda x: x['rotation_throughput'])
        print(f"• Rotation transformation: {slowest_rotation['rotation_throughput']:,.0f} Gaussians/sec")
        print(f"  (slowest with {slowest_rotation['n_gaussians']:,} Gaussians)")

    if 'depth_sort' in all_results:
        sort_results = all_results['depth_sort']
        slowest_sort = min(sort_results, key=lambda x: x['sort_throughput'])
        print(f"• Depth sorting: {slowest_sort['sort_throughput']:,.0f} Gaussians/sec")
        print(f"  (slowest with {slowest_sort['n_gaussians']:,} Gaussians)")

    if 'ply_io' in all_results:
        io_results = all_results['ply_io']
        slowest_write = min(io_results, key=lambda x: x['write_throughput'])
        print(f"• PLY write: {slowest_write['write_throughput']:,.0f} Gaussians/sec")
        print(f"  (slowest with {slowest_write['n_gaussians']:,} Gaussians)")

    print("\nRecommendations:")
    print("-" * 70)
    print("• For <10k Gaussians: All operations should be real-time")
    print("• For 10k-100k Gaussians: Depth sorting may become bottleneck")
    print("• For >100k Gaussians: Consider GPU-based sorting")
    print("• Rotation transformations are expensive - cache when possible")


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("Gaussian Splatting Performance Benchmark")
    print("=" * 70)

    # Test with various sizes
    n_gaussians_list = [100, 1_000, 10_000, 50_000, 100_000]

    print(f"\nTesting with: {[f'{n:,}' for n in n_gaussians_list]} Gaussians")

    all_results = {}

    # Run benchmarks
    try:
        all_results['ply_io'] = benchmark_ply_io(n_gaussians_list)
    except Exception as e:
        print(f"PLY I/O benchmark failed: {e}")

    try:
        all_results['rotation_transform'] = benchmark_coordinate_transforms(n_gaussians_list)
    except Exception as e:
        print(f"Coordinate transform benchmark failed: {e}")

    try:
        all_results['depth_sort'] = benchmark_depth_sorting(n_gaussians_list)
    except Exception as e:
        print(f"Depth sorting benchmark failed: {e}")

    try:
        all_results['rendering_utils'] = benchmark_rendering_utils(n_gaussians_list)
    except Exception as e:
        print(f"Rendering utils benchmark failed: {e}")

    # Print summary
    print_summary(all_results)

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
