# Covariance Matrix Transformation: Quick Reference

## TL;DR - Is It Correct?

**YES** ✓ The proposed transformation is **CORRECT**.

```python
P = np.array([[0, 0, 1],
              [0, 1, 0],
              [1, 0, 0]])
Σ_xyz = P @ Σ_zyx @ P.T
```

## One-Line Summary

Covariance matrices (second-order tensors) transform as **Σ' = P Σ P^T**, not as vectors (Σ' = P Σ).

---

## Quick Derivation

Given a linear transformation **y = Px**:

```
Σ_x = E[(x - μ_x)(x - μ_x)^T]
Σ_y = E[(Px - Pμ_x)(Px - Pμ_x)^T]
    = E[P(x - μ_x)(x - μ_x)^T P^T]
    = P Σ_x P^T
```

**Therefore: Σ_y = P Σ_x P^T** (congruence transformation)

---

## Why Not Just P Σ?

Covariance matrices have **two spatial indices**:

```
Σ_ij = E[(x_i - μ_i)(x_j - μ_j)]
```

When coordinates change, **both indices** transform:

```
Σ'_kl = Σ_ij (∂x_i/∂x'_k)(∂x_j/∂x'_l) = P_ki Σ_ij P_lj = (P Σ P^T)_kl
```

Vectors have one index → transform as **v' = Pv**
Covariances have two indices → transform as **Σ' = P Σ P^T**

---

## Is P Correct for ZYX → XYZ?

**YES** ✓

```python
P @ [z, y, x]^T = [x, y, z]^T

[[0, 0, 1],     [[z]     [[x]
 [0, 1, 0],  @   [y]  =   [y]
 [1, 0, 0]]      [x]]     [z]]
```

Row 1: selects index 2 (z → x)
Row 2: selects index 1 (y → y)
Row 3: selects index 0 (x → z)

**Result: (z,y,x) → (x,y,z)** ✓

---

## Properties Preserved

| Property | Preserved? | Why |
|----------|-----------|-----|
| **Symmetry** | ✓ Yes | (P Σ P^T)^T = P Σ^T P^T = P Σ P^T |
| **Positive definiteness** | ✓ Yes | v^T (P Σ P^T) v = (P^T v)^T Σ (P^T v) > 0 |
| **Determinant** | ✓ Yes | det(P Σ P^T) = det(Σ) · det(P)^2 = det(Σ) |
| **Eigenvalues** | ✓ Yes | Same eigenvalues (orthogonal transformation) |

---

## Test Cases

### Test 1: Diagonal Matrix
```python
Σ_zyx = [[1, 0, 0],
         [0, 4, 0],
         [0, 0, 9]]

Σ_xyz = [[9, 0, 0],
         [0, 4, 0],
         [0, 0, 1]]
```
**Result**: Diagonal elements swap positions (z↔x, y unchanged) ✓

### Test 2: With Correlations
```python
Σ_zyx = [[2.0, 0.5, 0.3],
         [0.5, 3.0, 0.7],
         [0.3, 0.7, 4.0]]

Σ_xyz = [[4.0, 0.7, 0.3],
         [0.7, 3.0, 0.5],
         [0.3, 0.5, 2.0]]
```
**Result**: Rows and columns permuted, symmetry preserved ✓

### Test 3: Roundtrip
```python
Σ → P @ Σ @ P.T → P @ (P @ Σ @ P.T) @ P.T → Σ
```
**Result**: Returns to original (P² = I) ✓

---

## Alternative: Index Permutation

For permutation matrices, these are **equivalent**:

```python
# Method 1: Matrix multiplication (general, clear)
Σ_xyz = P @ Σ_zyx @ P.T

# Method 2: Index permutation (faster, specific to permutations)
Σ_xyz = Σ_zyx[[2,1,0], :][:, [2,1,0]]
```

Both give **identical results**. Use Method 1 for clarity and generality.

---

## Common Mistakes

### ❌ Wrong: Forgetting P^T
```python
Σ_xyz = P @ Σ_zyx  # WRONG - loses symmetry!
```

### ❌ Wrong: Reversed order
```python
Σ_xyz = P.T @ Σ_zyx @ P  # WRONG - this is XYZ → ZYX (inverse)
```

### ✅ Correct
```python
Σ_xyz = P @ Σ_zyx @ P.T  # CORRECT
```

---

## Robust Implementation

```python
def transform_covariance_zyx_to_xyz(Σ_zyx, force_symmetric=True):
    """Transform covariance from ZYX to XYZ coordinates.

    Args:
        Σ_zyx: 3×3 covariance matrix in ZYX coordinates
        force_symmetric: Force output to be exactly symmetric

    Returns:
        Σ_xyz: 3×3 covariance matrix in XYZ coordinates
    """
    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype=Σ_zyx.dtype)

    Σ_xyz = P @ Σ_zyx @ P.T

    # Optional: force exact symmetry (eliminates floating-point errors)
    if force_symmetric:
        Σ_xyz = 0.5 * (Σ_xyz + Σ_xyz.T)

    return Σ_xyz
```

---

## Numerical Stability

### Good News
- Permutation is **exact** (no floating-point errors)
- Symmetry is **mathematically preserved**
- Positive definiteness is **guaranteed** (if input is PD)

### Optional Safeguards
```python
# Force symmetry (eliminate numerical noise)
Σ = 0.5 * (Σ + Σ.T)

# Ensure positive definiteness (if needed)
eigenvalues, eigenvectors = np.linalg.eigh(Σ)
eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
Σ = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# Check condition number (warn if ill-conditioned)
cond = np.linalg.cond(Σ)
if cond > 1e10:
    warnings.warn(f"Ill-conditioned covariance: {cond}")
```

---

## References

### Standard Textbooks
- **Hartley & Zisserman**, *Multiple View Geometry* (2003), Chapter 2
  - Covariance propagation in computer vision
- **Thrun, Burgard & Fox**, *Probabilistic Robotics* (2005), Eq. 3.15
  - Uncertainty propagation in robotics

### 3D Gaussian Splatting
- **Kerbl et al.**, *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023
  - Section 3.2: World-space covariance Σ = R S S^T R^T

### Classical Mechanics / Geometry
- **Abraham & Marsden**, *Foundations of Mechanics* (1978)
  - Tensor transformations under coordinate changes

---

## Summary

| Question | Answer |
|----------|--------|
| **Is the transformation correct?** | ✓ Yes |
| **Is P correct for ZYX→XYZ?** | ✓ Yes |
| **Why P Σ P^T not P Σ?** | Covariance is second-order tensor |
| **Symmetry preserved?** | ✓ Yes |
| **Positive definiteness preserved?** | ✓ Yes |
| **Numerically stable?** | ✓ Yes (exact permutation) |
| **Alternative formulation?** | Index permutation: Σ[[2,1,0],:][:,[2,1,0]] |
| **References?** | Hartley & Zisserman, Thrun et al., Kerbl et al. |

---

## Final Recommendation

**Use the proposed formula as-is**:

```python
P = np.array([[0, 0, 1],
              [0, 1, 0],
              [1, 0, 0]])
Σ_xyz = P @ Σ_zyx @ P.T
```

Optionally add `Σ_xyz = 0.5 * (Σ_xyz + Σ_xyz.T)` for robustness.

**This is mathematically correct and numerically stable.** ✓
