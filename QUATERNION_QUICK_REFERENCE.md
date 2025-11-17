# Quaternion Coordinate Transformation - Quick Reference

**For napari Gaussian Splatting Integration**

---

## TL;DR - What You Need to Know

### ❌ WRONG - Don't Do This
```python
# NEVER swap quaternion components directly!
quat_zyx = [x, y, z, w]
quat_xyz = [z, y, x, w]  # ❌ WRONG! Changes chirality!
```

### ✅ CORRECT - Do This Instead
```python
from scipy.spatial.transform import Rotation as R
import numpy as np

# Define permutation matrix
P = np.array([
    [0, 0, 1],  # new_X = old_Z
    [0, 1, 0],  # new_Y = old_Y
    [1, 0, 0]   # new_Z = old_X
])

# Transform quaternion
rot = R.from_quat(quat_zyx)  # scalar-last: [x,y,z,w]
R_matrix = rot.as_matrix()
R_new = P @ R_matrix @ P.T
quat_xyz = R.from_matrix(R_new).as_quat()
```

---

## Common Coordinate Transformations

### ZYX to XYZ (Swap first and last axes)
```python
P_zyx_to_xyz = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
])
```

### Y-up to Z-up (Common in graphics)
```python
# Method 1: Via rotation
coord_rotation = R.from_euler('x', -90, degrees=True)
rot_z_up = coord_rotation * rot_y_up * coord_rotation.inv()

# Method 2: Via matrix
P_y_to_z = R.from_euler('x', -90, degrees=True).as_matrix()
```

### Right-handed to Left-handed
```python
# Reflection in XY plane (negate Z)
P_flip_z = np.array([
    [1,  0,  0],
    [0,  1,  0],
    [0,  0, -1]
])
```

---

## Quaternion Format Cheat Sheet

| What | Format | Example |
|------|--------|---------|
| **SciPy (napari)** | scalar-last | `[x, y, z, w]` |
| **Gaussian Splatting** | often scalar-first | `[w, x, y, z]` |
| **VisPy** | use Quaternion class | `q.x, q.y, q.z, q.w` |

### Format Conversion
```python
def xyzw_to_wxyz(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]])

def wxyz_to_xyzw(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]])
```

---

## Complete Gaussian Transformation

```python
def transform_gaussian(pos, quat, scale, P):
    """
    Transform a 3D Gaussian to new coordinate system.

    Parameters
    ----------
    pos : array (3,)
        Position in old coordinates
    quat : array (4,)
        Rotation quaternion [x,y,z,w]
    scale : array (3,)
        Scale parameters
    P : array (3,3)
        Coordinate transformation matrix

    Returns
    -------
    pos_new, quat_new, scale_new : arrays
    """
    # Position
    pos_new = P @ pos

    # Rotation
    R_old = R.from_quat(quat).as_matrix()
    R_new = P @ R_old @ P.T
    quat_new = R.from_matrix(R_new).as_quat()

    # Scale
    scale_new = P @ scale

    return pos_new, quat_new, scale_new
```

---

## Edge Cases - Quick Checks

### 1. Sign Ambiguity
```python
# q and -q are the SAME rotation
def same_rotation(q1, q2):
    return np.allclose(q1, q2) or np.allclose(q1, -q2)
```

### 2. Unit Norm Check
```python
# Quaternions should always be unit length
assert np.allclose(np.linalg.norm(quat), 1.0)
```

### 3. Near-Identity
```python
# Check if rotation is ~zero
def is_identity(quat, tol_deg=0.1):
    angle = 2 * np.arccos(np.clip(abs(quat[3]), 0, 1))
    return np.rad2deg(angle) < tol_deg
```

### 4. 180° Rotation
```python
# For θ=180°, w≈0
def is_180_rotation(quat):
    return abs(quat[3]) < 1e-6
```

---

## Validation Quick Test

```python
def quick_validate(quat_old, quat_new, P):
    """Quick sanity check after transformation."""

    # Check 1: Unit norm
    assert np.allclose(np.linalg.norm(quat_new), 1.0), "Not unit quaternion!"

    # Check 2: Round-trip
    rot = R.from_quat(quat_new)
    R_back = P.T @ rot.as_matrix() @ P
    quat_back = R.from_matrix(R_back).as_quat()
    assert same_rotation(quat_old, quat_back), "Round-trip failed!"

    # Check 3: Determinant
    R_new = R.from_quat(quat_new).as_matrix()
    assert np.allclose(abs(np.linalg.det(R_new)), 1.0), "Bad determinant!"

    print("✓ All checks passed")
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Direct Component Swap
```python
# DON'T
quat_new = [quat[2], quat[1], quat[0], quat[3]]  # Changes chirality!
```

### ❌ Mistake 2: Ignoring Sign Ambiguity
```python
# DON'T
if quat1 == quat2:  # Might miss -quat2

# DO
if same_rotation(quat1, quat2):
```

### ❌ Mistake 3: Wrong Format
```python
# DON'T assume format without checking
quat_gs = load_gaussian_splatting_data()
rot = R.from_quat(quat_gs)  # Might be wxyz not xyzw!

# DO check and convert
if is_wxyz_format(quat_gs):
    quat_gs = wxyz_to_xyzw(quat_gs)
rot = R.from_quat(quat_gs)
```

### ❌ Mistake 4: Forgetting Scale
```python
# DON'T transform only position and rotation
pos_new = P @ pos
quat_new = transform_quat(quat, P)
# Scale components also need permutation!

# DO transform all components
scale_new = P @ scale  # Important!
```

### ❌ Mistake 5: Using P instead of P.T for inverse
```python
# DON'T
quat_back = transform_quat(quat_new, P)  # Wrong!

# DO
quat_back = transform_quat(quat_new, P.T)  # Correct
```

---

## Debugging Checklist

When transformations don't work:

- [ ] Check quaternion format (xyzw vs wxyz)
- [ ] Verify quaternion is unit norm
- [ ] Check permutation matrix is orthogonal (P @ P.T = I)
- [ ] Test with identity transformation (P = I)
- [ ] Verify round-trip (P then P.T)
- [ ] Check for sign flip (q vs -q)
- [ ] Validate with covariance matrix
- [ ] Test on simple cases (90°, 180° rotations)

---

## Performance Tips

### Batch Processing
```python
# Transform many Gaussians at once
def transform_gaussians_batch(positions, quats, scales, P):
    """Vectorized transformation."""
    n = len(positions)

    # Positions (vectorized)
    positions_new = (P @ positions.T).T

    # Rotations (loop unfortunately needed for quaternions)
    quats_new = np.zeros_like(quats)
    for i in range(n):
        R_old = R.from_quat(quats[i]).as_matrix()
        R_new = P @ R_old @ P.T
        quats_new[i] = R.from_matrix(R_new).as_quat()

    # Scales (vectorized)
    scales_new = (P @ scales.T).T

    return positions_new, quats_new, scales_new
```

### Caching for Same P
```python
# If transforming many Gaussians with same P, cache conversions
class CoordinateTransformer:
    def __init__(self, P):
        self.P = P
        self.P_T = P.T

    def transform(self, pos, quat, scale):
        pos_new = self.P @ pos
        R_old = R.from_quat(quat).as_matrix()
        R_new = self.P @ R_old @ self.P_T  # Use cached P.T
        quat_new = R.from_matrix(R_new).as_quat()
        scale_new = self.P @ scale
        return pos_new, quat_new, scale_new
```

---

## Integration with napari Code

### Existing napari quaternion utilities:
```python
# /home/user/napari/src/napari/_vispy/utils/quaternion.py
from napari._vispy.utils.quaternion import quaternion2euler_degrees

# Converts VisPy Quaternion to Euler angles
# Handles gimbal lock automatically
angles = quaternion2euler_degrees(vispy_quat)
```

### Existing camera code:
```python
# /home/user/napari/src/napari/components/camera.py
from scipy.spatial.transform import Rotation as R

# napari uses 'yzx' Euler angle sequence
rotation_matrix = R.from_euler(
    seq='yzx', angles=self.angles, degrees=True
).as_matrix()
```

---

## Resources

**Full Documentation:** `QUATERNION_RESEARCH_SUMMARY.md`
**Validation Code:** `quaternion_coordinate_transform_research.py`

**Key Links:**
- SciPy Rotation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
- Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
- Math Stack Exchange: https://math.stackexchange.com/questions/1795351/

---

## Quick Test Script

```python
# Copy-paste to test your implementation
import numpy as np
from scipy.spatial.transform import Rotation as R

# Test quaternion transformation
def test():
    # Create test quaternion (45° around X)
    quat = R.from_euler('x', 45, degrees=True).as_quat()

    # Permutation: (Z,Y,X) -> (X,Y,Z)
    P = np.array([[0,0,1], [0,1,0], [1,0,0]])

    # Transform
    R_old = R.from_quat(quat).as_matrix()
    R_new = P @ R_old @ P.T
    quat_new = R.from_matrix(R_new).as_quat()

    # Validate
    assert np.allclose(np.linalg.norm(quat_new), 1.0), "Not unit!"

    # Round-trip
    R_back = P.T @ R_new @ P
    quat_back = R.from_matrix(R_back).as_quat()
    assert (np.allclose(quat, quat_back) or
            np.allclose(quat, -quat_back)), "Round-trip failed!"

    print("✓ Tests passed!")

test()
```

---

**Remember:** When in doubt, verify with the covariance matrix: **Σ' = P @ Σ @ P^T**

This is the ultimate test that your transformation is correct!
