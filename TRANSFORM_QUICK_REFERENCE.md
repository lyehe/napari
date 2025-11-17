# Gaussian Splatting Transform Quick Reference

## Quick Start

```bash
# Run all tests
python test_gaussian_splatting_transforms.py

# Expected: ALL TESTS PASSED ✓
```

## Core Formulas

### Position Transform
```python
position_new = rotation @ scale @ position_old + translation
```

### Covariance Transform
```python
# Build covariance from rotation and scale
S = np.diag(scale)
covariance = rotation @ S @ S.T @ rotation.T

# Transform covariance
linear_matrix = transform_rotation @ transform_scale
covariance_new = linear_matrix @ covariance @ linear_matrix.T
```

### Rotation Conversions
```python
# Matrix to Quaternion [w, x, y, z]
from scipy.spatial.transform import Rotation as R
rot = R.from_matrix(rotation_matrix)
quat_xyzw = rot.as_quat()  # [x, y, z, w]
quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

# Quaternion to Matrix
rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert [w,x,y,z] to [x,y,z,w]
rotation_matrix = rot.as_matrix()
```

## Common Patterns

### Pattern 1: Create a Rotation
```python
# From Euler angles (degrees)
rotation = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()

# From axis-angle
axis = np.array([1, 1, 1]) / np.sqrt(3)  # Normalize
angle = np.deg2rad(45)
rotation = R.from_rotvec(angle * axis).as_matrix()

# 90° rotations
R_x_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
R_y_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
R_z_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
```

### Pattern 2: Validate a Rotation Matrix
```python
def is_valid_rotation(R):
    # Check orthogonality: R @ R.T = I
    is_orthogonal = np.allclose(R @ R.T, np.eye(3), atol=1e-8)

    # Check determinant: det(R) = +1
    is_proper = np.allclose(np.linalg.det(R), 1.0, atol=1e-8)

    return is_orthogonal and is_proper
```

### Pattern 3: Validate a Covariance Matrix
```python
def is_valid_covariance(cov):
    # Check symmetry
    is_symmetric = np.allclose(cov, cov.T, atol=1e-10)

    # Check positive definiteness
    eigenvalues = np.linalg.eigvals(cov)
    is_positive_definite = np.all(eigenvalues > 0)

    return is_symmetric and is_positive_definite
```

### Pattern 4: Compose Transformations
```python
# Compose two affine transforms
# T_combined = T2 @ T1 (apply T1 first, then T2)

R1, t1 = transform1_rotation, transform1_translation
R2, t2 = transform2_rotation, transform2_translation

R_combined = R2 @ R1
t_combined = R2 @ t1 + t2
```

### Pattern 5: Invert a Transformation
```python
# For rotation + translation
R_inv = R.T  # Transpose for rotation matrices
t_inv = -R_inv @ t

# For general linear transform
linear_inv = np.linalg.inv(linear_matrix)
t_inv = -linear_inv @ translation
```

### Pattern 6: Extract Components from Affine Matrix
```python
# From 4x4 homogeneous matrix
affine_4x4 = ...

rotation = affine_4x4[:3, :3]  # Top-left 3x3
translation = affine_4x4[:3, 3]  # Top-right 3x1

# Decompose rotation + scale
from scipy.linalg import qr
rotation_only, upper_tri = qr(rotation)
scale = np.diag(upper_tri)
```

## Test Checklist

When implementing new transformation code:

- [ ] Test identity transform (should do nothing)
- [ ] Test 90° rotations (exact values, easy to verify)
- [ ] Test uniform scaling (multiply by constant)
- [ ] Test roundtrip (forward then inverse should recover original)
- [ ] Verify rotation orthogonality (R @ R.T = I)
- [ ] Verify covariance symmetry (Σ = Σ.T)
- [ ] Verify covariance positive definite (eigenvalues > 0)
- [ ] Test with extreme values (1e-10, 1e10)
- [ ] Test numerical stability (100+ sequential operations)

## Common Mistakes

### ❌ Wrong: Transform covariance with single matrix
```python
covariance_new = transform @ covariance  # WRONG!
```

### ✅ Right: Use similarity transformation
```python
covariance_new = transform @ covariance @ transform.T  # CORRECT
```

---

### ❌ Wrong: Compare quaternions directly
```python
assert np.allclose(quat1, quat2)  # WRONG! (double cover: q = -q)
```

### ✅ Right: Compare rotation matrices
```python
R1 = quaternion_to_matrix(quat1)
R2 = quaternion_to_matrix(quat2)
assert np.allclose(R1, R2)  # CORRECT
```

---

### ❌ Wrong: Wrong transformation order
```python
result = translation + rotation @ position  # WRONG!
```

### ✅ Right: Apply rotation before translation
```python
result = rotation @ position + translation  # CORRECT
```

---

### ❌ Wrong: Forget to normalize quaternion
```python
quat = [2, 0, 0, 0]  # Not normalized!
rotation = quaternion_to_matrix(quat)  # Will give wrong result
```

### ✅ Right: Always normalize
```python
quat = [2, 0, 0, 0]
quat = quat / np.linalg.norm(quat)  # Normalize first
rotation = quaternion_to_matrix(quat)  # CORRECT
```

---

### ❌ Wrong: Use too strict tolerance
```python
# After 100 operations:
assert np.allclose(result, expected, rtol=1e-15)  # Will fail!
```

### ✅ Right: Use appropriate tolerance
```python
# After 100 operations:
assert np.allclose(result, expected, rtol=1e-8, atol=1e-8)  # CORRECT
```

## Tolerance Guidelines

| Operation | Relative Tol | Absolute Tol |
|-----------|--------------|--------------|
| Single transform | 1e-10 | 1e-10 |
| Roundtrip (2 ops) | 1e-9 | 1e-9 |
| 10 operations | 1e-8 | 1e-8 |
| 100 operations | 1e-7 | 1e-7 |
| Comparison with analytical | 1e-12 | 1e-12 |

## Debugging Tips

### Problem: Gaussians appear stretched/distorted
**Check**: Is covariance symmetric and positive definite?
```python
print("Symmetric:", np.allclose(cov, cov.T))
print("Eigenvalues:", np.linalg.eigvals(cov))  # Should all be > 0
```

### Problem: Rotation looks wrong
**Check**: Is rotation matrix orthogonal?
```python
print("Orthogonality error:", np.linalg.norm(R @ R.T - np.eye(3)))
print("Determinant:", np.linalg.det(R))  # Should be +1
```

### Problem: Roundtrip doesn't recover original
**Check**: Is transformation invertible?
```python
print("Condition number:", np.linalg.cond(transform))  # < 1e10 is good
print("Determinant:", np.linalg.det(transform))  # Should be != 0
```

### Problem: Numerical instability
**Check**: Are values in reasonable range?
```python
print("Scale range:", np.min(scale), "-", np.max(scale))
# Ideally between 1e-6 and 1e6
print("Position range:", np.min(positions), "-", np.max(positions))
# Should be reasonable for scene
```

## Performance Tips

### Tip 1: Vectorize position transforms
```python
# Slow: Loop
for i in range(n):
    new_positions[i] = transform @ positions[i] + translation

# Fast: Vectorize
new_positions = (transform @ positions.T).T + translation
```

### Tip 2: Cache rotation matrices
```python
# If quaternion doesn't change, cache the matrix
if self._quat_cached != quat:
    self._rotation_cached = quaternion_to_matrix(quat)
    self._quat_cached = quat.copy()
rotation = self._rotation_cached
```

### Tip 3: Avoid repeated inverse
```python
# Compute once, reuse
transform_inv = np.linalg.inv(transform)

# Use multiple times
for gaussian in gaussians:
    gaussian.inverse_transform(transform_inv, ...)
```

### Tip 4: Use in-place operations
```python
# Allocate once
result = np.empty_like(positions)

# Compute in-place
np.matmul(transform, positions.T, out=result.T)
result += translation
```

## Useful Invariants

These should always be true:

1. **Rotation orthogonality**: `R @ R.T = I`
2. **Rotation determinant**: `det(R) = +1`
3. **Quaternion norm**: `‖q‖ = 1`
4. **Covariance symmetry**: `Σ = Σ.T`
5. **Covariance positive**: `eigenvalues(Σ) > 0`
6. **Trace invariance**: `trace(R @ Σ @ R.T) = trace(Σ)`
7. **Volume scaling**: `det(Σ') = det(transform)² × det(Σ)`

## Quick Checks

```python
# Rotation matrix
assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
assert np.allclose(np.linalg.det(R), 1.0, atol=1e-8)

# Quaternion
assert np.allclose(np.linalg.norm(q), 1.0, atol=1e-8)

# Covariance
assert np.allclose(Σ, Σ.T, atol=1e-10)
assert np.all(np.linalg.eigvals(Σ) > 0)

# Roundtrip
forward = transform @ position + translation
backward = transform_inv @ (forward - translation)
assert np.allclose(backward, position, atol=1e-9)
```

## Example: Complete Gaussian Transform

```python
def transform_gaussian(gaussian, rotation, scale, translation):
    """Transform a Gaussian splat."""

    # 1. Build transformation matrix
    transform_matrix = rotation @ np.diag(scale)

    # 2. Transform position
    new_position = transform_matrix @ gaussian.position + translation

    # 3. Transform covariance
    new_covariance = (transform_matrix @
                     gaussian.covariance @
                     transform_matrix.T)

    # 4. Validate result
    assert np.allclose(new_covariance, new_covariance.T, atol=1e-10)
    assert np.all(np.linalg.eigvals(new_covariance) > 0)

    # 5. Extract new rotation and scale via eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(new_covariance)
    new_scale = np.sqrt(np.abs(eigenvalues))
    new_rotation = eigenvectors

    # Ensure proper rotation
    if np.linalg.det(new_rotation) < 0:
        new_rotation[:, -1] *= -1

    return GaussianSplat(
        position=new_position,
        rotation=new_rotation,
        scale=new_scale,
        opacity=gaussian.opacity,
        color=gaussian.color
    )
```

## File Reference

| File | Purpose |
|------|---------|
| `test_gaussian_splatting_transforms.py` | Main test suite (31 tests) |
| `test_gaussian_splatting_transforms_guide.md` | Detailed documentation |
| `example_gaussian_transform_usage.py` | Practical examples |
| `TEST_RESULTS_SUMMARY.md` | Test results and analysis |
| `TRANSFORM_QUICK_REFERENCE.md` | This file |

## Resources

- **napari transforms**: `/home/user/napari/src/napari/utils/transforms/`
- **scipy.spatial.transform**: https://docs.scipy.org/doc/scipy/reference/spatial.transform.html
- **3D Gaussian Splatting paper**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

**Last updated**: 2025-11-17
**Version**: 1.0
