# Quaternion Coordinate Transformation Research Summary

**Research Date:** 2025-11-17
**Context:** napari Gaussian Splatting integration
**Branch:** claude/gaussian-splatting-napari-plan-01WV7p14WXQi76ycS5cBqjF3

---

## Executive Summary

Quaternions **DO** require transformation when changing coordinate systems, but **NOT** through direct component swapping. The correct approach uses rotation matrix transformation: **R' = P @ R @ P^T**, where P is the coordinate transformation (permutation) matrix.

### Key Findings

1. ✅ **Correct Method**: Convert to rotation matrix, apply permutation, convert back
2. ❌ **Wrong Method**: Direct quaternion component swapping (changes chirality)
3. ⚠️  **Critical**: Account for quaternion sign ambiguity (q and -q are equivalent)
4. ⚠️  **Format**: SciPy uses scalar-last (x,y,z,w), Gaussian Splatting often uses scalar-first (w,x,y,z)

---

## 1. Mathematical Proof of Quaternion Transformation

### The Problem

When changing coordinate systems (e.g., from ZYX to XYZ), we need to transform all geometric quantities, including rotations represented as quaternions.

### Theoretical Foundation

**Quaternions represent rotations, not coordinate system axes.** A rotation that transforms vector **v** to **v'** in one coordinate system must be re-expressed to achieve the equivalent transformation in a different coordinate system.

Given:
- A rotation represented by quaternion **q** in coordinate system A
- A coordinate transformation from system A to system B represented by permutation matrix **P**
- We want to find quaternion **q'** that represents the same physical rotation in system B

### The Mathematics

#### Step 1: Quaternion to Rotation Matrix
Convert quaternion **q = (x, y, z, w)** to rotation matrix **R**:

```
For unit quaternion q = w + xi + yj + zk:

R = [1-2(y²+z²)   2(xy-wz)    2(xz+wy)  ]
    [2(xy+wz)     1-2(x²+z²)  2(yz-wx)  ]
    [2(xz-wy)     2(yz+wx)    1-2(x²+y²)]
```

#### Step 2: Transform the Rotation Matrix
Apply the coordinate transformation:

```
R' = P @ R @ P^T
```

**Proof of correctness:**
If **v** is a vector in system A and **v_B = P @ v** is the same vector in system B, then:
- In system A: **R @ v** gives the rotated vector
- In system B: **P @ (R @ v)** is the rotated vector expressed in B
- We want: **R' @ v_B = R' @ (P @ v) = P @ (R @ v)**
- Therefore: **R' = P @ R @ P^{-1} = P @ R @ P^T** (since P is orthogonal)

This proves that **P @ R @ P^T** correctly transforms the rotation to the new coordinate system.

#### Step 3: Convert Back to Quaternion
Extract quaternion **q'** from rotation matrix **R'** using standard algorithms (e.g., Shepperd's method in SciPy).

### Why Direct Component Swapping Fails

Swapping quaternion components (e.g., [x, y, z, w] → [z, y, x, w]) is equivalent to:
1. Permuting the basis vectors in the quaternion algebra
2. This changes the **chirality** (handedness) of the coordinate system
3. Results in incorrect rotation that doesn't preserve the physical transformation

**Mathematical proof:** The quaternion [x, y, z, w] represents rotation by angle θ around axis [x, y, z]/sin(θ/2). Simply permuting components changes the rotation axis without accounting for how that axis relates to the new coordinate frame.

---

## 2. Alternative Methods

### Method 1: Rotation Matrix Transformation (Recommended)

**Advantages:**
- Mathematically rigorous
- Works for any coordinate transformation
- Easy to verify correctness

**Implementation:**
```python
from scipy.spatial.transform import Rotation as R
import numpy as np

def transform_quaternion_via_matrix(quat_in, P):
    """
    Transform quaternion using rotation matrix method.

    Parameters
    ----------
    quat_in : array (4,)
        Input quaternion in scalar-last format [x, y, z, w]
    P : array (3, 3)
        Coordinate transformation matrix

    Returns
    -------
    quat_out : array (4,)
        Transformed quaternion
    """
    rot = R.from_quat(quat_in)
    R_matrix = rot.as_matrix()
    R_transformed = P @ R_matrix @ P.T
    return R.from_matrix(R_transformed).as_quat()
```

**Example - ZYX to XYZ transformation:**
```python
# Permutation matrix: (Z,Y,X) -> (X,Y,Z)
P = np.array([
    [0, 0, 1],  # new_X = old_Z
    [0, 1, 0],  # new_Y = old_Y
    [1, 0, 0]   # new_Z = old_X
])

quat_xyz = transform_quaternion_via_matrix(quat_zyx, P)
```

### Method 2: Quaternion Composition

**Advantages:**
- More intuitive for axis-aligned transformations
- Avoids matrix conversions if you have the coordinate rotation as quaternion

**Implementation:**
```python
def transform_quaternion_via_composition(quat_in, coord_rotation):
    """
    Transform quaternion using quaternion composition.

    Parameters
    ----------
    quat_in : array (4,)
        Input quaternion
    coord_rotation : Rotation object
        Rotation representing coordinate system change

    Returns
    -------
    quat_out : array (4,)
        Transformed quaternion
    """
    rot_in = R.from_quat(quat_in)
    # Conjugation: R' = Q * R * Q^(-1)
    rot_out = coord_rotation * rot_in * coord_rotation.inv()
    return rot_out.as_quat()
```

**Example - Y-up to Z-up:**
```python
# Coordinate change: rotate frame by -90° around X
coord_rotation = R.from_euler('x', -90, degrees=True)
quat_z_up = transform_quaternion_via_composition(quat_y_up, coord_rotation)
```

### Method 3: Direct Formula for Specific Permutations

For pure axis permutations without reflections, there exist direct formulas, but they are error-prone and not recommended unless performance is absolutely critical.

---

## 3. Edge Cases and Special Considerations

### Edge Case 1: Quaternion Sign Ambiguity (Double Cover)

**Issue:** Quaternions **q** and **-q** represent the **identical** rotation.

**Mathematical basis:** Unit quaternions form a double cover of SO(3), meaning the map from quaternions to rotations is 2:1.

**Implications:**
- When comparing quaternions, must check both **q** and **-q**
- When averaging quaternions, ensure all are on same hemisphere
- When interpolating (SLERP), choose shortest path

**Detection and handling:**
```python
def quaternions_equivalent(q1, q2, atol=1e-6):
    """Check if two quaternions represent the same rotation."""
    return np.allclose(q1, q2, atol=atol) or np.allclose(q1, -q2, atol=atol)

def enforce_quaternion_continuity(q_current, q_previous):
    """Ensure quaternion has same sign as previous for smooth interpolation."""
    if np.dot(q_current, q_previous) < 0:
        return -q_current
    return q_current
```

### Edge Case 2: Gimbal Lock

**Key Insight:** Gimbal lock is **NOT** a problem for quaternions themselves!

**Details:**
- Gimbal lock occurs in Euler angle representations
- Quaternions are singularity-free for representing rotations
- **However**: Converting between quaternions and Euler angles can encounter singularities
- napari's `quaternion2euler_degrees()` handles this with epsilon checks

**Implication:** When possible, avoid converting to Euler angles. Stick with quaternions or rotation matrices.

**Example from napari codebase:**
```python
# From /home/user/napari/src/napari/_vispy/utils/quaternion.py
epsilon = 1e-10
sin_theta_2 = 2 * (q.w * q.y - q.z * q.x)

if abs(sin_theta_2) > 1 - epsilon:
    # Gimbal lock detected - handle specially
    theta_1 = -np.sign(sin_theta_2) * 2 * np.arctan2(q.x, q.w)
    theta_2 = np.arcsin(sin_theta_2)
    theta_3 = 0  # Not uniquely determined
```

### Edge Case 3: Near-Zero Rotations

**Issue:** Very small rotations can have numerical instabilities.

**Characteristics:**
- Near-identity rotation: **q ≈ [0, 0, 0, 1]**
- The axis becomes ill-defined as angle → 0
- Quaternions handle this gracefully (unlike axis-angle)

**Handling:**
```python
def is_near_identity(quat, angle_threshold_deg=0.1):
    """Check if quaternion represents near-zero rotation."""
    # For small angles θ: w ≈ cos(θ/2) ≈ 1 - θ²/8
    # So: angle ≈ 2 * acos(w)
    angle_rad = 2 * np.arccos(np.clip(abs(quat[3]), 0, 1))
    return np.rad2deg(angle_rad) < angle_threshold_deg
```

### Edge Case 4: 180-Degree Rotations

**Issue:** For θ = 180°, w = 0, and axis determination from quaternion becomes ambiguous.

**Characteristics:**
- **q = [x, y, z, 0]** where **(x, y, z)** defines the rotation axis
- Multiple quaternions can represent same 180° rotation about axis
- Sign of axis vector matters: **[x,y,z,0]** and **[-x,-y,-z,0]** are different rotations!

**Handling:**
```python
def is_180_degree_rotation(quat, threshold=1e-6):
    """Check if quaternion represents ~180° rotation."""
    return abs(quat[3]) < threshold  # w ≈ 0

def get_180_rotation_axis(quat):
    """Extract axis for 180° rotation (assumes w ≈ 0)."""
    axis = quat[:3]
    return axis / np.linalg.norm(axis)
```

### Edge Case 5: Quaternion Format Mismatch

**Critical Issue:** Different libraries use different quaternion orderings!

| Library/Format | Order | Example |
|---------------|-------|---------|
| SciPy | scalar-last | [x, y, z, w] |
| Gaussian Splatting (some) | scalar-first | [w, x, y, z] |
| Unity | scalar-last | [x, y, z, w] |
| Eigen (C++) | scalar-first | [w, x, y, z] |
| ROS tf | scalar-last | [x, y, z, w] |

**Detection and conversion:**
```python
def convert_quat_format(quat, from_format='xyzw', to_format='wxyz'):
    """Convert between quaternion formats."""
    if from_format == to_format:
        return quat

    if from_format == 'xyzw' and to_format == 'wxyz':
        return np.array([quat[3], quat[0], quat[1], quat[2]])
    elif from_format == 'wxyz' and to_format == 'xyzw':
        return np.array([quat[1], quat[2], quat[3], quat[0]])
    else:
        raise ValueError(f"Unsupported conversion: {from_format} -> {to_format}")
```

---

## 4. Gaussian Splatting Specific Considerations

### Coordinate Transformation Pipeline

For a complete Gaussian transformation, you must transform:

1. **Position** (3D vector)
2. **Rotation** (quaternion)
3. **Scale** (3D vector)
4. **Spherical Harmonics** (if present)

### Complete Transformation Code

```python
def transform_gaussian_to_new_coords(gaussian_params, P):
    """
    Transform a 3D Gaussian to a new coordinate system.

    Parameters
    ----------
    gaussian_params : dict
        Dictionary with keys:
        - 'position': (3,) array
        - 'quaternion': (4,) array in scalar-last format
        - 'scale': (3,) array
        - 'sh_coeffs': (N, 3) array (optional, for colored Gaussians)
    P : array (3, 3)
        Coordinate transformation matrix

    Returns
    -------
    transformed_params : dict
        Gaussian parameters in new coordinate system
    """
    # 1. Transform position
    position_new = P @ gaussian_params['position']

    # 2. Transform rotation (via rotation matrix)
    rot = R.from_quat(gaussian_params['quaternion'])
    R_matrix = rot.as_matrix()
    R_new = P @ R_matrix @ P.T
    quaternion_new = R.from_matrix(R_new).as_quat()

    # 3. Transform scale (permute components)
    scale_new = P @ gaussian_params['scale']

    # 4. Transform spherical harmonics (if present)
    sh_new = None
    if 'sh_coeffs' in gaussian_params:
        sh_new = transform_spherical_harmonics(
            gaussian_params['sh_coeffs'],
            R_new
        )

    return {
        'position': position_new,
        'quaternion': quaternion_new,
        'scale': scale_new,
        'sh_coeffs': sh_new
    }

def transform_spherical_harmonics(sh_coeffs, R_matrix):
    """
    Transform SH coefficients using Wigner D-matrix.

    Note: This is a placeholder. Full implementation requires:
    - Wigner D-matrix computation for SH orders 2 and 3
    - Libraries: e3nn, sphecerix, or custom implementation

    References:
    - https://github.com/graphdeco-inria/gaussian-splatting/issues/176
    """
    # Order 0 (DC component): no rotation needed
    # Order 1: direct 3x3 rotation
    # Order 2+: need Wigner D-matrix

    # Simplified for order 1 only:
    if sh_coeffs.shape[0] == 3:  # Only first-order SH
        return R_matrix @ sh_coeffs
    else:
        raise NotImplementedError(
            "Full SH rotation requires Wigner D-matrix. "
            "See: https://github.com/graphdeco-inria/gaussian-splatting/issues/176"
        )
```

### Verification via Covariance Matrix

The covariance matrix provides an excellent way to verify transformations:

**Formula:** Σ = R S S^T R^T

```python
def verify_gaussian_transformation(params_old, params_new, P):
    """
    Verify Gaussian transformation by checking covariance matrix.

    Returns
    -------
    bool
        True if transformation is correct
    """
    # Compute covariance in old system
    R_old = R.from_quat(params_old['quaternion']).as_matrix()
    S_old = np.diag(params_old['scale'])
    cov_old = R_old @ S_old @ S_old.T @ R_old.T

    # Compute covariance in new system
    R_new = R.from_quat(params_new['quaternion']).as_matrix()
    S_new = np.diag(params_new['scale'])
    cov_new = R_new @ S_new @ S_new.T @ R_new.T

    # Transform old covariance directly
    cov_transformed = P @ cov_old @ P.T

    # They should match!
    return np.allclose(cov_new, cov_transformed)
```

---

## 5. Links to Authoritative Sources

### Primary References

1. **SciPy Rotation Documentation**
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
   - Official documentation for quaternion handling
   - Describes scalar-last format [x, y, z, w]
   - Euler angle conventions

2. **3D Gaussian Splatting Paper**
   https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
   arXiv: https://arxiv.org/abs/2308.04079
   - Original paper describing covariance parameterization
   - Formula: Σ = R S S^T R^T

3. **Gaussian Splatting GitHub Issues**
   https://github.com/graphdeco-inria/gaussian-splatting/issues/176
   - Detailed discussion on rotating Gaussians
   - Explains wxyz vs xyzw format difference
   - Describes Wigner D-matrix for spherical harmonics

### Mathematical Background

4. **Quaternions and Spatial Rotation (Wikipedia)**
   https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
   - Comprehensive mathematical background
   - Double cover property explained
   - Conversion formulas

5. **Axis Permutations with Quaternions**
   https://math.stackexchange.com/questions/1795351/axes-permutations-and-negations-using-quaternions
   - Mathematical Stack Exchange discussion
   - Explains reflection operations
   - Levi-Civita tensor approach

6. **Coordinate System Transformation**
   https://math.stackexchange.com/questions/727793/how-can-i-transform-coordinate-systems-with-quaternions
   - Practical guidance on quaternion composition
   - Order of operations clarified

7. **Why Can't You Swap XYZ in Quaternions?**
   https://stackoverflow.com/questions/16099979/can-i-switch-x-y-z-in-a-quaternion
   - Explains chirality problem
   - Alternative approaches discussed

### Advanced Topics

8. **Quaternion Double Cover**
   https://www.reedbeta.com/blog/why-quaternions-double-cover/
   Nathan Reed's excellent explanation of the double cover property

9. **Gimbal Lock and Quaternions**
   https://en.wikipedia.org/wiki/Gimbal_lock
   - Why Euler angles have singularities
   - Why quaternions don't

10. **Quaternions in Iowa State Course Notes (2024)**
    https://faculty.sites.iastate.edu/jia/files/inline-files/quaternion.pdf
    - Recent academic treatment
    - Detailed mathematical proofs

### Tutorials and Implementations

11. **Gaussian Splatting Explained (Towards Data Science)**
    https://towardsdatascience.com/a-comprehensive-overview-of-gaussian-splatting-e7d570081362
    - Overview of the method

12. **Python Engineer's Guide to Gaussian Splatting Part 2**
    https://towardsdatascience.com/a-python-engineers-introduction-to-3d-gaussian-splatting-part-2-7e45b270c1df
    - Practical implementation details
    - Quaternion usage explained

13. **How to Render a Single Gaussian Splat**
    https://shi-yan.github.io/how_to_render_a_single_gaussian_splat/
    - Step-by-step mathematical breakdown
    - Covariance matrix construction

14. **Gaussian Splatting Formulas (kwea123)**
    https://github.com/kwea123/gaussian_splatting_notes
    - Detailed formula explanations
    - Rotation matrix from quaternion

---

## 6. Validation Checklist

When implementing quaternion coordinate transformations:

### Before Implementation
- [ ] Identify the source and target coordinate systems
- [ ] Determine the permutation matrix P
- [ ] Check quaternion format (scalar-first vs scalar-last)
- [ ] Verify if spherical harmonics need transformation

### During Implementation
- [ ] Use rotation matrix method: R' = P @ R @ P^T
- [ ] Transform all components (position, rotation, scale)
- [ ] Preserve quaternion unit norm
- [ ] Handle sign ambiguity when comparing

### After Implementation
- [ ] Verify round-trip transformation (P then P^T)
- [ ] Check covariance matrix: Σ' = P @ Σ @ P^T
- [ ] Test on known cases (identity, 90°, 180° rotations)
- [ ] Validate with multiple random rotations
- [ ] Check edge cases (near-zero, 180°, gimbal lock angles)

### Validation Code Template

```python
def validate_transformation(transform_func, P):
    """Run comprehensive validation tests."""

    tests = []

    # Test 1: Unit norm preservation
    q = R.random().as_quat()
    q_transformed = transform_func(q, P)
    tests.append(np.allclose(np.linalg.norm(q_transformed), 1.0))

    # Test 2: Round-trip
    q_roundtrip = transform_func(q_transformed, P.T)
    tests.append(quaternions_equivalent(q, q_roundtrip))

    # Test 3: Identity transformation
    I = np.eye(3)
    q_identity = transform_func(q, I)
    tests.append(quaternions_equivalent(q, q_identity))

    # Test 4: Covariance consistency
    scale = np.random.rand(3) + 0.1
    R_mat = R.from_quat(q).as_matrix()
    S = np.diag(scale)
    cov = R_mat @ S @ S.T @ R_mat.T

    R_mat_new = R.from_quat(q_transformed).as_matrix()
    scale_new = P @ scale
    S_new = np.diag(scale_new)
    cov_new = R_mat_new @ S_new @ S_new.T @ R_mat_new.T

    cov_expected = P @ cov @ P.T
    tests.append(np.allclose(cov_new, cov_expected))

    return all(tests), tests
```

---

## 7. Recommendations for napari Integration

### Architecture Recommendations

1. **Use SciPy Rotation consistently**
   - Standardize on scalar-last format [x, y, z, w]
   - Avoid manual quaternion arithmetic
   - Let SciPy handle conversions

2. **Create utility module**
   ```python
   # napari/layers/utils/coordinate_transforms.py

   def transform_quaternion(quat, coord_transform_matrix):
       """Standard quaternion transformation for napari."""
       # Implementation using rotation matrix method
       pass

   def transform_gaussian_params(params, coord_transform_matrix):
       """Transform all Gaussian parameters together."""
       # Ensures consistency across position, rotation, scale
       pass
   ```

3. **Validation in tests**
   - Add comprehensive tests in `test_coordinate_transforms.py`
   - Include edge case tests
   - Use property-based testing (hypothesis) for random rotations

4. **Documentation**
   - Document coordinate system conventions clearly
   - Warn about quaternion format differences
   - Provide examples for common transformations

### Performance Considerations

For performance-critical code:
- Cache rotation matrices if transforming many Gaussians with same P
- Use vectorized operations for batch transformations
- Consider numba JIT compilation for hot paths

```python
import numba as nb

@nb.jit(nopython=True, cache=True)
def transform_quaternions_batch(quats, P):
    """Transform batch of quaternions efficiently."""
    # Vectorized implementation
    pass
```

### Integration with existing napari code

The existing `/home/user/napari/src/napari/_vispy/utils/quaternion.py` provides:
- `quaternion2euler_degrees()` - handles gimbal lock correctly

Could be extended with:
- `transform_quaternion_coords()` - for coordinate transformations
- `quaternions_equivalent()` - for comparison with sign handling
- `enforce_quaternion_continuity()` - for interpolation

---

## 8. Code Examples Summary

All code examples are provided in:
- **Implementation:** `/home/user/napari/quaternion_coordinate_transform_research.py`
- **This document:** Sections 2-4 above

Key examples:
1. Rotation matrix transformation method
2. Quaternion composition method
3. Gaussian Splatting complete pipeline
4. Edge case handling
5. Validation framework

Run the research script:
```bash
python quaternion_coordinate_transform_research.py
```

Expected output: All 5 validation tests pass ✓

---

## 9. Conclusion

### What We Learned

1. **Quaternion transformation IS necessary** when changing coordinate systems
2. **The correct method** uses rotation matrix transformation: R' = P @ R @ P^T
3. **Direct component swapping FAILS** because it changes chirality
4. **Edge cases matter**: sign ambiguity, format differences, 180° rotations
5. **Gaussian Splatting requires** transforming position, rotation, scale, and SH

### Critical Implementation Points

- ✅ Use `scipy.spatial.transform.Rotation` for all quaternion operations
- ✅ Transform via rotation matrices, not component manipulation
- ✅ Verify with covariance matrix: Σ' = P @ Σ @ P^T
- ✅ Handle quaternion sign ambiguity in comparisons
- ✅ Check quaternion format (xyzw vs wxyz)
- ⚠️ Spherical harmonics need Wigner D-matrix rotation (separate implementation)

### Next Steps for napari

1. Implement `transform_quaternion()` utility function
2. Add comprehensive test suite with edge cases
3. Document coordinate system conventions
4. Consider Spherical Harmonics support for full Gaussian Splatting
5. Validate against reference implementations

---

## References

See Section 5 for complete list of 14+ authoritative sources including:
- SciPy documentation
- 3D Gaussian Splatting paper and GitHub
- Mathematics Stack Exchange discussions
- Academic course notes
- Tutorial articles

**All validation tests passed:** 5/5 ✓

---

*Generated: 2025-11-17*
*Research validation script: quaternion_coordinate_transform_research.py*
