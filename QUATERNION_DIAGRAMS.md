# Quaternion Coordinate Transformation - Visual Diagrams

**ASCII diagrams and visual explanations**

---

## 1. Coordinate System Transformation Overview

```
Original System (ZYX)              Target System (XYZ)
      Z↑                                  Z↑
       |                                  |
       |                                  |
       |____→ Y                           |____→ Y
      /                                  /
     / X                                / X


Same physical space, different axis labels!
We need to transform the rotation representation.
```

---

## 2. Why Direct Component Swapping Fails

```
❌ WRONG APPROACH:
Original quat in ZYX: [x, y, z, w]
                       ↓  ↓  ↓  ↓
                       swap components
                       ↓  ↓  ↓  ↓
New quat in XYZ:      [z, y, x, w]  ← INCORRECT!

Problem: This changes the CHIRALITY (handedness)
```

### Example Visualization:

```
Original rotation: 45° around X-axis in ZYX system
Quaternion: [0.38, 0, 0, 0.92]
            └─┬─┘        └──┬─┘
              X            cos(θ/2)

After wrong swap: [0, 0, 0.38, 0.92]
                      └──┬─┘
                         Z
This represents 45° around Z-axis - WRONG!
```

---

## 3. Correct Rotation Matrix Method

```
✅ CORRECT APPROACH:

Step 1: Quaternion → Rotation Matrix
   [x,y,z,w] ──→  R (3×3)

Step 2: Transform the matrix
   R' = P × R × P^T

   where P is the coordinate transformation:
   P = [0 0 1]   (X_new = Z_old)
       [0 1 0]   (Y_new = Y_old)
       [1 0 0]   (Z_new = X_old)

Step 3: Rotation Matrix → Quaternion
   R' ──→ [x',y',z',w']
```

### Mathematical Flow:

```
Physical Rotation
      ↓
Quaternion in System A: [x, y, z, w]
      ↓
Rotation Matrix in A: R_A
      ↓
Transform: R_B = P × R_A × P^T
      ↓
Rotation Matrix in B: R_B
      ↓
Quaternion in System B: [x', y', z', w']
      ↓
Same Physical Rotation!
```

---

## 4. Permutation Matrix Visualization

### ZYX to XYZ Permutation:

```
Input vector in ZYX order: [Z, Y, X]
                            ↓  ↓  ↓
Permutation Matrix P:      [0, 0, 1]  ← takes X (3rd position)
                           [0, 1, 0]  ← takes Y (2nd position)
                           [1, 0, 0]  ← takes Z (1st position)
                            ↓  ↓  ↓
Output vector in XYZ:      [X, Y, Z]


Matrix multiplication view:
[0 0 1]   [Z]   [X]
[0 1 0] × [Y] = [Y]
[1 0 0]   [X]   [Z]
```

### Common Permutations:

```
Identity (no change):       Cyclic X→Y→Z→X:        Swap X↔Z:
[1 0 0]                     [0 1 0]                [0 0 1]
[0 1 0]                     [0 0 1]                [0 1 0]
[0 0 1]                     [1 0 0]                [1 0 0]
```

---

## 5. Gaussian Splatting Covariance Visualization

### Covariance Construction:

```
Gaussian Parameters:
   Position: p (3×1)
   Rotation: q (4×1) → R (3×3)
   Scale:    s (3×1) → S (3×3 diagonal)

Covariance Matrix Construction:
   Σ = R × S × S^T × R^T

Visual breakdown:
   S = [s₁  0   0 ]     S×S^T = [s₁²  0    0  ]
       [0   s₂  0 ]             [0    s₂²  0  ]
       [0   0   s₃]             [0    0    s₃²]

   R rotates the ellipsoid axes
   S scales along each axis
```

### Transformation:

```
Original System:               New System:
   Σ = R S S^T R^T               Σ' = R' S' S'^T R'^T

Should satisfy:
   Σ' = P × Σ × P^T

This is the KEY VERIFICATION!
```

### Visual Example:

```
Original Gaussian in ZYX:
       Z
       ↑
       |  ╱
       | ╱ ← elongated
       |╱
   ────┼──→ Y
      ╱|
     ╱ |
    X  |

Scale: [s_z=2, s_y=0.5, s_x=0.3]
Rotation: 90° around Z

Transformed to XYZ:
       Z
       ↑
       | ╱
       |╱ ← same physical ellipsoid
   ────┼──→ Y
      ╱|
     ╱ |
    X  |

Scale: [s_x=0.3, s_y=0.5, s_z=2]  ← permuted!
Rotation: 90° around X  ← transformed!
```

---

## 6. Quaternion Sign Ambiguity (Double Cover)

```
Unit Quaternion Sphere (4D, projected to 3D):

      +q
       *
      /|\
     / | \
    /  |  \
   *───●───*  ← Equator (w=0, 180° rotations)
    \  |  /
     \ | /
      \|/
       *
      -q

Property: +q and -q represent SAME rotation!

Example:
  q  = [ 0.38, 0, 0,  0.92] ─┐
                              ├─ Same 45° rotation around X
  -q = [-0.38, 0, 0, -0.92] ─┘

When comparing: must check both!
```

### Interpolation Issue:

```
Interpolating between q₁ and q₂:

Wrong (might take long path):
q₁ ───────────────────────→ q₂
    (could be >180° arc)

Correct (check dot product):
if dot(q₁, q₂) < 0:
    q₂ = -q₂  ← flip to same hemisphere

q₁ ─────→ q₂
  (short path, <90° arc)
```

---

## 7. Edge Case: Gimbal Lock (Euler Angles vs Quaternions)

### Euler Angles (Problematic):

```
3D Rotation Space with Euler Angles:

     Pitch
       ↑
       |
   +90°├─────── ← SINGULARITY! (Gimbal Lock)
       |
       |
       └──────→ Yaw
      /
     /
  Roll

At pitch = ±90°:
- Roll and Yaw rotations become equivalent
- Loss of one degree of freedom
- Euler → Quaternion → Euler gives different angles!
```

### Quaternions (No Gimbal Lock):

```
Unit Quaternion Space (3-sphere in 4D):

      [0,0,0,1]
         * ← Identity rotation
        /|\
       / | \  ← No singularities!
      *──●──*    Smooth everywhere
       \ | /
        \|/
         *

Properties:
✓ No singularities
✓ Smooth interpolation
✓ Compact representation
✗ Sign ambiguity (q = -q)
```

---

## 8. Transformation Decision Tree

```
START: Need to transform rotation to new coordinates?
  |
  ├─ Is it a simple axis-aligned permutation?
  |   ├─ YES → Use rotation matrix method
  |   |         (Convert to matrix, apply P, convert back)
  |   |
  |   └─ NO → Is it a rotation of the coordinate frame?
  |             ├─ YES → Use quaternion composition method
  |             |         (Compose with frame rotation)
  |             |
  |             └─ COMPLEX → Use rotation matrix method
  |                         (Most general approach)
  |
  ├─ After transformation:
  |   ├─ Verify unit norm: ‖q‖ = 1
  |   ├─ Check round-trip: P then P^T
  |   └─ Validate covariance: Σ' = P Σ P^T
  |
  └─ DONE ✓
```

---

## 9. Batch Transformation Pipeline

```
Input: N Gaussians in System A
  |
  ├─ Positions (N×3)
  |   └─→ Vectorized: P @ positions.T → positions_new
  |
  ├─ Quaternions (N×4)
  |   └─→ Loop (unfortunately):
  |       for each q:
  |         R = quat_to_matrix(q)
  |         R' = P @ R @ P^T
  |         q' = matrix_to_quat(R')
  |
  ├─ Scales (N×3)
  |   └─→ Vectorized: P @ scales.T → scales_new
  |
  └─ Spherical Harmonics (N×K×3) [optional]
      └─→ Requires Wigner D-matrix rotation
          (Advanced: see Gaussian Splatting issue #176)
  |
Output: N Gaussians in System B ✓
```

---

## 10. Common Permutation Matrices Reference

```
┌────────────────────────────────────────────────────┐
│  ZYX → XYZ (Swap first and last)                  │
│  [0 0 1]                                          │
│  [0 1 0]                                          │
│  [1 0 0]                                          │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│  Cyclic: XYZ → YZX                                │
│  [0 1 0]                                          │
│  [0 0 1]                                          │
│  [1 0 0]                                          │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│  Reflection: Flip Z (right → left handed)         │
│  [1  0  0]                                        │
│  [0  1  0]                                        │
│  [0  0 -1]                                        │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│  Y-up → Z-up (via 90° rotation around X)          │
│  [1   0   0]                                      │
│  [0   0  -1]                                      │
│  [0   1   0]                                      │
└────────────────────────────────────────────────────┘
```

---

## 11. Validation Flow Chart

```
                    ┌─────────────────┐
                    │  Transform      │
                    │  Quaternion     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Check Unit Norm │
                    │   ‖q'‖ = 1 ?    │
                    └────┬──────┬─────┘
                         │ YES  │ NO → ERROR!
                    ┌────▼──────▼─────┐
                    │  Round-Trip     │
                    │  P then P^T     │
                    │  → back to q ?  │
                    └────┬──────┬─────┘
                         │ YES  │ NO → ERROR!
                    ┌────▼──────▼─────┐
                    │  Covariance     │
                    │  Σ' = PΣP^T ?   │
                    └────┬──────┬─────┘
                         │ YES  │ NO → ERROR!
                    ┌────▼──────▼─────┐
                    │ Random Tests    │
                    │ (Monte Carlo)   │
                    └────┬──────┬─────┘
                         │ PASS │ FAIL → DEBUG!
                         ▼      ▼
                    ┌─────────────────┐
                    │   VALIDATED ✓   │
                    └─────────────────┘
```

---

## 12. Format Conversion Diagram

```
Different Quaternion Formats:

SciPy / NumPy:                 Gaussian Splatting:
┌──────────────┐              ┌──────────────┐
│ x  y  z  w   │              │ w  x  y  z   │
│ ↑  ↑  ↑  ↑   │              │ ↑  ↑  ↑  ↑   │
│ imaginary│   │              │ │  imaginary │
│         real │              │ real         │
└──────────────┘              └──────────────┘
 scalar-last                   scalar-first

Conversion:
[x, y, z, w] ←→ [w, x, y, z]
     ↓                ↓
  [3,0,1,2] ←→ [1,2,3,0]  (index permutation)

⚠️  ALWAYS CHECK FORMAT BEFORE USING! ⚠️
```

---

## 13. Physical Interpretation

```
Same Physical Gaussian in Different Coordinate Systems:

System ZYX (old):                    System XYZ (new):
┌──────────────────────┐            ┌──────────────────────┐
│ Position: [1, 2, 3]  │            │ Position: [3, 2, 1]  │
│                      │            │                      │
│ Rotation: describes  │   SAME     │ Rotation: describes  │
│   how ellipsoid is   │  OBJECT!   │   how ellipsoid is   │
│   oriented relative  │   ─────→   │   oriented relative  │
│   to ZYX axes        │            │   to XYZ axes        │
│                      │            │                      │
│ Scale: [2, 0.5, 0.3] │            │ Scale: [0.3, 0.5, 2] │
│   along Z, Y, X      │            │   along X, Y, Z      │
└──────────────────────┘            └──────────────────────┘

The NUMBERS change, but the PHYSICAL SHAPE is identical!
```

---

## 14. Summary Diagram: Correct vs Wrong

```
╔════════════════════════════════════════════════════════════╗
║  QUATERNION COORDINATE TRANSFORMATION                      ║
╚════════════════════════════════════════════════════════════╝

WRONG ❌                          CORRECT ✅
──────────────────────          ──────────────────────
Direct swap:                    Via rotation matrix:

q = [x, y, z, w]               q = [x, y, z, w]
     ↓ ↓ ↓ ↓                        ↓
Permute indices                 to_matrix()
     ↓ ↓ ↓ ↓                        ↓
q'= [z, y, x, w]               R = 3×3 matrix
     ⚠️                             ↓
Changes chirality!              R' = P @ R @ P^T
Incorrect rotation!                 ↓
                               from_matrix()
                                    ↓
                               q' = [x', y', z', w']
                                    ✓
                               Correct transformation!

╔════════════════════════════════════════════════════════════╗
║  VERIFICATION: Σ' = P @ Σ @ P^T  (covariance check)       ║
╚════════════════════════════════════════════════════════════╝
```

---

## 15. Complete Pipeline Visualization

```
┌──────────────────────────────────────────────────────────────┐
│           GAUSSIAN SPLATTING COORDINATE TRANSFORM            │
└──────────────────────────────────────────────────────────────┘

INPUT (System A):
┌───────────┬────────────┬────────────┬──────────────┐
│ Position  │ Quaternion │   Scale    │ Sph. Harm.   │
│   (3×1)   │   (4×1)    │   (3×1)    │   (K×3)      │
└─────┬─────┴──────┬─────┴──────┬─────┴──────┬───────┘
      │            │            │            │
      │ P@p        │ R'=PRP^T   │ P@s        │ Wigner-D
      ↓            ↓            ↓            ↓
┌─────────────────────────────────────────────────────┐
│ TRANSFORMATION MATRIX P (coordinate change)         │
└─────────────────────────────────────────────────────┘
      ↓            ↓            ↓            ↓
      │            │            │            │
      ↓            ↓            ↓            ↓
┌─────┴─────┬──────┴─────┬──────┴─────┬──────┴───────┐
│ Position' │Quaternion' │  Scale'    │   SH'        │
│   (3×1)   │   (4×1)    │   (3×1)    │   (K×3)      │
└───────────┴────────────┴────────────┴──────────────┘

OUTPUT (System B):

VERIFY: Covariance Σ' = P @ Σ @ P^T ✓
```

---

## ASCII Art: 3D Rotation Visualization

```
Unit Quaternion Sphere (Conceptual 3D Projection):

           w=1 (identity)
              ★
             /│\
            / │ \
           /  │  \          All points on this sphere
          /   │   \         represent valid rotations
         /    │    \
        /     │     \
       │      │      │
    ───┼──────●──────┼───  w=0 plane (180° rotations)
       │             │
        \           /
         \         /
          \       /
           \     /
            \   /
             \ /
              ★
           w=-1 (also identity!)

Every rotation has TWO representations: ±q
```

---

**End of Diagrams**

For implementation details, see:
- `QUATERNION_RESEARCH_SUMMARY.md` - Full mathematical background
- `QUATERNION_QUICK_REFERENCE.md` - Copy-paste code snippets
- `quaternion_coordinate_transform_research.py` - Executable validation
