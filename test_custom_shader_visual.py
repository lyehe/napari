"""Test script for custom Gaussian shader visual."""

import numpy as np


def test_gaussian_visual_creation():
    """Test that GaussianMarkers visual can be created."""
    try:
        from napari._vispy.visuals.gaussian_splatting import GaussianMarkers

        visual = GaussianMarkers()
        print("✓ GaussianMarkers visual created successfully")

        # Check that VBOs are created
        assert hasattr(visual, '_vbo_pos'), "Missing position VBO"
        assert hasattr(visual, '_vbo_rot'), "Missing rotation VBO"
        assert hasattr(visual, '_vbo_scale'), "Missing scale VBO"
        assert hasattr(visual, '_vbo_opacity'), "Missing opacity VBO"
        assert hasattr(visual, '_vbo_color'), "Missing color VBO"
        print("✓ All VBOs created")

        # Check that shader program has correct attributes
        assert 'a_position' in visual.shared_program, "Missing a_position attribute"
        assert 'a_rotation' in visual.shared_program, "Missing a_rotation attribute"
        assert 'a_scale' in visual.shared_program, "Missing a_scale attribute"
        assert 'a_opacity' in visual.shared_program, "Missing a_opacity attribute"
        assert 'a_color' in visual.shared_program, "Missing a_color attribute"
        print("✓ All shader attributes registered")

        # Check that uniforms are initialized
        assert 'u_point_size' in visual.shared_program, "Missing u_point_size uniform"
        assert 'u_viewport' in visual.shared_program, "Missing u_viewport uniform"
        print("✓ All shader uniforms initialized")

        return True
    except Exception as e:
        print(f"✗ Failed to create visual: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gaussian_data_setting():
    """Test setting Gaussian data."""
    try:
        from napari._vispy.visuals.gaussian_splatting import GaussianMarkers

        visual = GaussianMarkers()

        # Create test data
        n = 10
        positions = np.random.rand(n, 3).astype(np.float32)
        rotations = np.tile([0, 0, 0, 1], (n, 1)).astype(np.float32)
        scales = np.ones((n, 3), dtype=np.float32)
        opacities = np.ones(n, dtype=np.float32)
        colors = np.random.rand(n, 3).astype(np.float32)

        # Set data
        visual.set_gaussian_data(
            positions=positions,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
        )

        print("✓ Gaussian data set successfully")

        # Check that data is stored
        assert visual._n_gaussians == n, f"Expected {n} Gaussians, got {visual._n_gaussians}"
        assert visual._positions is not None, "Positions not stored"
        assert visual._rotations is not None, "Rotations not stored"
        assert visual._scales is not None, "Scales not stored"
        assert visual._opacities is not None, "Opacities not stored"
        assert visual._colors is not None, "Colors not stored"
        print("✓ All data stored correctly")

        # Check data shapes
        assert visual._positions.shape == (n, 3), f"Wrong positions shape: {visual._positions.shape}"
        assert visual._rotations.shape == (n, 4), f"Wrong rotations shape: {visual._rotations.shape}"
        assert visual._scales.shape == (n, 3), f"Wrong scales shape: {visual._scales.shape}"
        assert visual._opacities.shape == (n,), f"Wrong opacities shape: {visual._opacities.shape}"
        assert visual._colors.shape == (n, 3), f"Wrong colors shape: {visual._colors.shape}"
        print("✓ All data shapes correct")

        return True
    except Exception as e:
        print(f"✗ Failed to set data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gaussian_splatting_visual():
    """Test the full GaussianSplattingVisual."""
    try:
        from napari._vispy.visuals.gaussian_splatting import GaussianSplattingVisual

        visual = GaussianSplattingVisual()
        print("✓ GaussianSplattingVisual created successfully")

        # Check that it has the gaussian_markers component
        assert hasattr(visual, 'gaussian_markers'), "Missing gaussian_markers"
        assert hasattr(visual, 'selection_markers'), "Missing selection_markers"
        print("✓ Visual has all components")

        # Create test data
        n = 10
        positions = np.random.rand(n, 3).astype(np.float32)
        rotations = np.tile([0, 0, 0, 1], (n, 1)).astype(np.float32)
        scales = np.ones((n, 3), dtype=np.float32)
        opacities = np.ones(n, dtype=np.float32)
        colors = np.random.rand(n, 3).astype(np.float32)

        # Set data
        visual.set_data(
            positions=positions,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
            point_size_multiplier=1.5,
        )
        print("✓ Data set on GaussianSplattingVisual")

        return True
    except Exception as e:
        print(f"✗ Failed with GaussianSplattingVisual: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scene_node_wrapper():
    """Test that scene node wrapper works."""
    try:
        from napari._vispy.visuals.gaussian_splatting import GaussianMarkersNode

        # Create node
        node = GaussianMarkersNode()
        print("✓ GaussianMarkersNode scene wrapper created")

        # Check that it has the set_gaussian_data method
        assert hasattr(node, 'set_gaussian_data'), "Scene node missing set_gaussian_data method"
        print("✓ Scene node has set_gaussian_data method")

        return True
    except Exception as e:
        print(f"✗ Failed with scene node: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing Custom Gaussian Shader Visual")
    print("=" * 70)

    tests = [
        ("Visual Creation", test_gaussian_visual_creation),
        ("Data Setting", test_gaussian_data_setting),
        ("Compound Visual", test_gaussian_splatting_visual),
        ("Scene Node Wrapper", test_scene_node_wrapper),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'Test: ' + name:-<70}")
        result = test_func()
        results.append((name, result))

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
