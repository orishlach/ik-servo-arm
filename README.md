# adaptive-hand-prototype <br> Kinematics

### Task 1: Extend the implementation of the FK function to any number of joints.
<details>
<summary> 
1. different link lengths
</summary>
 The original implementation assumed identical link lengths for all joints. I modified the `SimpleArm` constructor to accept different link lengths:
 <figure>
    <img src="./screenshots/1.1-link-lengths.png" alt="Alt Text" width="550"/><br>
  <figcaption>Example configuration with different link lengths: [1.4, 0.5, 1.0]</figcaption>
</figure>
</details>
<details>
<summary> 
2. forward kinematics 
</summary>
The forward kinematics computation was generalized to support any number of joints using an iterative approach. I modified the `FK` function to calculate FK for n-links:
 <figure>
    <img src="./screenshots/1.2-n-joint-fk.png" alt="Alt Text" width="550"/><br>
  <figcaption>Example configuration with 4 links with lengths: [1.4, 0.5, 1.0,0.5]</figcaption>
</figure>
</details>

### Task 2: Gradient descent based IK (the Jacobian transpose method)

<details>
<summary> 
1. Implementation of `VelocityJacobian` method
</summary>
Each column of the Jacobian represents the instantaneous velocity of the end-effector when the corresponding joint moves at unit velocity.

Mathematically,for each joint i , the column:

$J{[:,i]}$ 

is given by:

$J{[:,i]} = Z_i\times(P_{end} - P_i)$

**Visualization of the Jacobian:**

<figure>
    <video width="550" controls>
      <source src="./screenshots/2.1-arrows.mp4" type="video/mp4">
    </video>
<figcaption>
Arrows appear as tangents to a circle based on the selected joint.

Additionally, their size changes according to the result of the Jacobian calculation.
</figcaption>
</figure>
</details>

<details>
<summary> 
`IK` method
</summary>

- The algorithm performs with 1000 steps and alpha=0.01

- **In areas farther away:**  
  The manipulator encounters challenges such as:  
  - Loss of degrees of freedom, making adjustments harder.  
  - The target being physically out of reach.  
  - The linear gradient descent approximation being less effective for nonlinear behavior.  

<figure>
    <video width="550" controls>
      <source src="./screenshots/2.2-gradient.mp4" type="video/mp4">
    </video>
<figcaption>
    The gradient value shown in the graph reflects the optimization process. It performs reasonably well but does not achieve the desired level of accuracy or speed.
</figcaption>
</details>


### Task 3: Gauss-Newton based IK (the Jacobian inverse method)

<details>
<summary> 
1. Implementation of `IK` method with 'Gauss-Newton'
</summary>

The Gauss-Newton method uses the pseudo-inverse of the Jacobian:

$\theta_{new} = \theta_{old} + \alpha J^+(x_{target} - x_{current})$

Where $J^+$ is the Moore-Penrose pseudo-inverse:

$J^+ = J^T(JJ^T)^{-1}$

<figure>
    <video width="550" controls>
      <source src="./screenshots/3-gauss-newton.mp4" type="video/mp4">
    </video>
<figcaption>
  The graph showing the gradient or error value demonstrates how Gauss-Newton converges significantly faster compared to gradient descent. However, small fluctuations or a lack of perfect convergence may still be visible near the target due to numerical limitations or sensitivity to local Jacobian behavior.
</figcaption>
</figure>
</details>

<details>
<summary> 
2. Extra degrees of freedom
</summary>

To define a unique solution for redundant robots, one approach is to use weights on the joint movements.

By assigning a cost or weight to each joint, the algorithm can prioritize certain joints over others, minimizing unnecessary movement.

This can be done by adding a weighted term to the optimization objective, such as minimizing energy, joint limits, or a desired posture, guiding the system to a preferred configuration.
</figcaption>
</details>
