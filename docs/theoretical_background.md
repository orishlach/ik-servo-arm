# Theoretical & Mathematical Background

## 1. Introduction
Robotic arms serve as central components in industrial automation, research prototyping, and educational labs. In this project, we focus on a **4-DOF** robotic arm comprising:

1. Two **revolute** (rotational) joints (often called R),
2. Two **prismatic** (linear) joints (often called P).

We refer to such a configuration as a **2R2P** arm. The following sections detail the main theoretical concepts:

- **Forward Kinematics (FK)**: Determining the end-effector's position from the joint variables.
- **Inverse Kinematics (IK)**: Finding the joint variables needed to reach a desired end-effector target.
- **Jacobian-Based Methods**: Explaining how partial derivatives (Jacobian) inform velocity relations and numerical IK.
- **Weighted Cost Functions**: Balancing end-effector accuracy with practical constraints (e.g., avoiding large angles or excessive extensions).

---

## 2. Kinematics

### 2.1 Forward Kinematics (FK)

**Forward Kinematics** answers: "Given the joint variables θ₁, θ₂, δ₁, δ₂, where does the end-effector end up in 3D space?"

We define the state vector:

```
X = [θ₁, θ₂, δ₁, δ₂]ᵀ
```

where:
- θ₁, θ₂∈ [0,π] are revolute joint angles (in radians),
- δ₁, δ₂∈ [0,4.5] are prismatic joint extensions.

Let each revolute link i have a nominal length ℓᵢ. If the link can also extend by δᵢ, its effective length becomes ℓᵢ + δᵢ. For a **planar** arm visualized in 3D (rotation only around the z-axis, translation in the x-y plane), the end-effector position p_end(X) can be computed by accumulating transformations:

```
p₁ = p₀ + Rz(θ₁) [ℓ₁ + δ₁, 0, 0]ᵀ
p₂ = p₁ + Rz(θ₁ + θ₂) [ℓ₂ + δ₂, 0, 0]ᵀ
```

with p₀ = [0, 0, 0]ᵀ as the base, and

```
Rz(θ) = [cos(θ) -sin(θ) 0;
         sin(θ)  cos(θ) 0;
         0       0      1]
```

Thus, the end-effector position is p_end = p₂. Although most motion may be in the x-y plane, we treat it as a 3D vector for completeness.

---

### 2.2 Inverse Kinematics (IK)

**Inverse Kinematics** answers: "Given a desired end-effector location p_target, how do we solve for θ₁, θ₂, δ₁, δ₂?"

#### 2.2.1 Cost Function & Weights

When a closed-form solution is cumbersome—especially with a 2R2P configuration—we use **numerical optimization**. We define a **cost function** C(X) that incorporates:

1. **Accuracy**: 
   ‖p_end(X) - p_target‖²
2. **Angle Regularization**: 
   w_θ(θ₁² + θ₂²)
3. **Prismatic Regularization**: 
   w_δ(δ₁² + δ₂²)

Hence,

```
C(X) = w_θ(θ₁² + θ₂²) + w_δ(δ₁² + δ₂²) + ‖p_end(X) - p_target‖²
```

- A **higher** w_θ discourages large angular displacements (keeping the arm more compact).
- A **higher** w_δ discourages large prismatic extensions.

Balancing these weights is a **design choice**:
- If prismatic extensions are mechanically weak, **increase** w_δ.
- If servo rotation beyond a certain angle might cause collisions, **increase** w_θ.

#### 2.2.2 Numerical Optimization

A simple workflow might be:

1. **Compute the gradient** ∇C(X).
2. **Determine a step direction** via the Jacobian or a pseudo-Newton approach (e.g., J^T·J + diagonal terms).
3. **Perform a line search** to find step size α.
4. **Update** X ← X - α · (step direction).
5. **Clamp** X so θᵢ and δⱼ remain within physical limits.

Stop when the **gradient norm** is sufficiently small or a maximum iteration count is reached.

---

## 3. Jacobian-Based Methods

### 3.1 Velocity Jacobian

A **Jacobian matrix** J(X) relates joint velocities Ẋ to the end-effector's velocity ṗ_end:

```
ṗ_end = J(X)·Ẋ
```

For a **2R2P** arm, J ∈ ℝ³ˣ⁴. We often split it into:

- J_θ ∈ ℝ³ˣ² (revolute partial derivatives),
- J_δ ∈ ℝ³ˣ² (prismatic partial derivatives),

and then horizontally concatenate to form the full J. Numerically:

```
J(X) = [J_θ J_δ]
```

### 3.2 Using the Jacobian for IK

In the **cost function** gradient, we have a term proportional to

```
J(X)ᵀ(p_end(X) - p_target)
```

for the positional error. Additionally, the angle and extension penalties yield:

```
2·w_θ[θ₁, θ₂, 0, 0]ᵀ and 2·w_δ[0, 0, δ₁, δ₂]ᵀ
```

Hence, the total gradient might look like:

```
∇C(X) = 2·J(X)ᵀ(p_end - p_target) + 2·w_θ[θ₁, θ₂, 0, 0]ᵀ + 2·w_δ[0, 0, δ₁, δ₂]ᵀ
```

This is used in numerical solvers (gradient descent, Gauss-Newton, etc.) to iteratively adjust X.

---

## 4. Significance of Weighting Terms

### 4.1 Physical Rationale

1. **Servo or Hardware Limits**  
   - Large θ can stress servo motors or cause collisions.  
   - Large δ can create instability or bending in the linear rails.

2. **Task-Specific Preferences**  
   - Some tasks favor minimal motion or a compact posture (high weights).  
   - Others favor rapid or unconstrained movement (lower weights).

### 4.2 Tuning Strategy

- **Start small**: w_θ = 0, w_δ = 0 → pure positional accuracy.
- **Observe results**: If the solver extends too far or angles become extreme, incrementally increase weights.
- **Iterate**: Adjust until final configurations respect mechanical safety and still reach the target acceptably.

---

## 5. Additional Details

### 5.1 Clamping & Limits

Each iteration must clamp:

```
θᵢ ∈ [0, π], δᵢ ∈ [0, 4.5]
```

This ensures the solver never sends invalid commands.

### 5.2 Handling Singularities

- **Singularities** occur if J loses rank (e.g., collinear joints).
- Near singularities, small joint changes may drastically move the end-effector, or the solver might fail to converge smoothly.
- The cost function plus weight penalties can reduce the chance of entering poorly-conditioned regions if they significantly raise the penalty.

### 5.3 Real-Time Control

If the arm moves continuously (e.g., following a path), you might:
- Update X at each time step using a feedback controller.
- Dynamically adjust w_θ, w_δ if you detect mechanical issues (like high current draw or near-limit extension).

---

## 6. Example Equations Recap

### 6.1 Forward Kinematics (Matrix Form)

```
p₁ = Rz(θ₁)[ℓ₁ + δ₁, 0, 0]ᵀ
p₂ = p₁ + Rz(θ₁ + θ₂)[ℓ₂ + δ₂, 0, 0]ᵀ
p_end = p₂
```

### 6.2 Cost Function with Weights

```
C(X) = w_θ(θ₁² + θ₂²) + w_δ(δ₁² + δ₂²) + ‖p_end(X) - p_target‖²
```

### 6.3 Gradient (Simplified)

```
∇C(X) = 2·J(X)ᵀ(p_end - p_target) + 2·w_θ[θ₁, θ₂, 0, 0]ᵀ + 2·w_δ[0, 0, δ₁, δ₂]ᵀ
```

---

## 7. Conclusion

By combining:

1. **Forward Kinematics** (to map joint values to end-effector position),
2. **Inverse Kinematics** (to solve for joint values given a target),
3. **Jacobian Methods** (for efficient gradient or pseudo-inverse updates), and
4. **Weighted Cost Functions** (to incorporate mechanical constraints and preferences),

we achieve a practical and robust approach for controlling a 2R2P robotic arm. Proper **clamping** of joint values, careful **tuning** of weights, and awareness of **singularities** ensure safe, reliable operation. This framework underpins the Python code (including `Rot()`, `IK()`, `update_physical_arm()`), the GUI sliders, and the Pololu Maestro servo interface in our overall project.