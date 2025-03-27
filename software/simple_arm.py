import vedo as vd
import numpy as np
from vedo.pyplot import plot

from pololu_maestro import PololuMaestro

###############################################################################
# Utility Functions
###############################################################################
def line_search(func, X, d, max_ls_iters=10):
    alpha = 1.0
    iteration = 0
    while func(X + alpha*d) > func(X):
        alpha *= 0.5
        iteration += 1
        if iteration >= max_ls_iters or alpha < 1e-6:
            break
    return alpha

def Rot(angle, axis):
    """
    Returns a 3x3 rotation matrix for rotating 'angle' radians about 'axis'.
    """
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    I = np.eye(3)
    K = np.array([
        [0,         -axis[2],  axis[1]],
        [axis[2],   0,         -axis[0]],
        [-axis[1],  axis[0],   0      ]
    ])
    R = I + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
    return R

###############################################################################
# SimpleArm Class
###############################################################################
class SimpleArm:
    """
    This arm has 2 revolute joints and 2 prismatic joints:
      X = [theta1, theta2, delta1, delta2].

    Indices:
      - X[0], X[1]: angles in radians
      - X[2], X[3]: prismatic extensions in [0..4.5] (example)

    Also contains a PololuMaestro object to move real servos whenever angles change.

    We do NOT call update_physical_arm() inside FK() to avoid hammering the hardware.
    Instead, we call update_physical_arm() either from sliders or after IK.
    """
    def __init__(self, maestro_port="COM3", maestro_baud=9600):
        # 2 revolve, 2 prismatic => total of 4
        self.X = np.zeros(4)     # [theta1, theta2, delta1, delta2]
        self.n = 2               # number of revolute joints
        self.L = [14.0, 14.0]    # link lengths (for revolve portion)
        self.Jw = np.zeros((self.n + 1, 3))  # 3D positions of the joints

        # Weights for IK
        self.WeightDelta = 0.0
        self.WeightAngle = 0.0

        # Connect to the Maestro (optional hardware connection)
        self.maestro = PololuMaestro(port=maestro_port, baudrate=maestro_baud)

        # Initial forward kinematics
        self.FK()

    def FK(self, X=None):
        """
        Forward Kinematics: compute 3D positions of the revolve joints.
        """
        if X is not None:
            self.X = X

        angles = self.X[:2]
        deltas = self.X[2:]
        self.Jw[0, :] = [0, 0, 0]
        total_angle = 0.0

        for i in range(1, self.n + 1):
            total_angle += angles[i - 1]
            R = Rot(total_angle, [0, 0, 1])

            # clamp extension to 4.5 max
            length_i = min(self.L[i - 1] + deltas[i - 1], 4.5)
            self.Jw[i, :] = self.Jw[i - 1, :] + R @ np.array([length_i, 0, 0])

        return self.Jw[-1, :]

    def velocity_jacobian(self, X=None):
        """
        Returns J_theta, J_d (3 x 2 each), for revolve + prismatic.
        """
        if X is not None:
            self.X = X

        angles = self.X[:2]
        z_axis = np.array([0, 0, 1])
        p_end = self.Jw[-1, :]
        J_theta = np.zeros((3, 2))
        J_d = np.zeros((3, 2))
        total_angle = 0.0

        for i in range(2):
            p_i = self.Jw[i, :]
            J_theta[:, i] = np.cross(z_axis, p_end - p_i)

            total_angle += angles[i]
            R = Rot(total_angle, z_axis)
            J_d[:, i] = R @ np.array([1, 0, 0])  # partial derivative w.r.t. prismatic extension

        return J_theta, J_d

    def IK(self, target):
        """
        Minimize:
          WeightAngle * ||angles||^2
          + WeightDelta * ||deltas||^2
          + ||end-effector - target||^2
        using a simple pseudo-Newton approach with line-search.
        """
        max_iterations = 1000
        w1 = self.WeightDelta
        w2 = self.WeightAngle

        def objective_value(vecX):
            oldX = self.X.copy()
            self.FK(vecX)
            err_vec = self.Jw[-1, :] - target
            a = vecX[:2]
            d = vecX[2:]
            cost = w2 * np.sum(a**2) + w1 * np.sum(d**2) + np.sum(err_vec**2)
            self.FK(oldX)
            return cost

        gradient_norms = []

        for i in range(max_iterations):
            current_pos = self.FK()
            error = current_pos - target

            angles = self.X[:2]
            deltas = self.X[2:]
            J_theta, J_d = self.velocity_jacobian()
            J = np.hstack((J_theta, J_d))

            # Weighted gradient
            a_ext = np.hstack([angles, np.zeros(2)])
            d_ext = np.hstack([np.zeros(2), deltas])
            grad = J.T @ error + 2*w2*a_ext + 2*w1*d_ext

            # Approximate Hessian
            H_a = 2*w2*np.eye(2)
            H_d = 2*w1*np.eye(2)
            H = np.block([
                [H_a, np.zeros((2, 2))],
                [np.zeros((2, 2)), H_d]
            ])
            JTJ = J.T @ J
            A = JTJ + H
            A_inv = np.linalg.pinv(A)
            dX = -(A_inv @ grad)

            x_current = self.X.copy()
            alpha = line_search(objective_value, x_current, dX, max_ls_iters=10)
            self.X += alpha * dX

            # Clamp angles to [0, pi], deltas to [0..4.5]
            self.X[0] = np.clip(self.X[0], 0, np.pi)
            self.X[1] = np.clip(self.X[1], 0, np.pi)
            self.X[2] = np.clip(self.X[2], 0.0, 4.5)
            self.X[3] = np.clip(self.X[3], 0.0, 4.5)

            gn = np.linalg.norm(grad)
            gradient_norms.append(gn)
            if gn < 1e-6:
                print(f"Converged in {i} iterations.")
                break

        final_deg = np.degrees(self.X[:2])
        final_deltas = self.X[2:]
        print("Final angles (deg):", ", ".join([f"{fd:.2f}" for fd in final_deg]))
        print("Final deltas:", ", ".join([f"{fd:.2f}" for fd in final_deltas]))

        # Update hardware
        self.update_physical_arm()

        return gradient_norms

    def update_physical_arm(self):
        """
        Send commands to 4 Maestro channels:
          - Channels 0..1: revolve joints (angles)
          - Channels 2..3: prismatic joints (gear extension)
        """
        if not self.maestro or not self.maestro.ser or not self.maestro.ser.is_open:
            return

        # revolve angles in radians
        revolve_angles = self.X[:2].copy()
        # offset for servo #1 (demo):
        revolve_angles[1] += np.radians(90)

        # prismatic
        prismatic = self.X[2:]

        # Channels 0..1 => revolve angles
        for ch in range(2):
            angle_deg = np.degrees(revolve_angles[ch])
            # expand or scale angle if needed
            angle_deg_expanded = angle_deg

            # Map 0..180° => 500..2500 µs
            us = 500 + (angle_deg_expanded / 180.0) * 2000
            target_q = int(us * 4)
            print(
                f"[Revolve Servo {ch}] "
                f"Raw angle (deg)={angle_deg:.2f}, "
                f"Expanded angle (deg)={angle_deg_expanded:.2f}, "
                f"us={us:.1f}, target_q={target_q}"
            )
            self.maestro.set_target(ch, target_q)

        # Channels 2..3 => prismatic gear extension
        for i in range(2):
            extension = prismatic[i]
            us_prism = 500 + (extension / 4.5) * 2000
            target_q_prism = int(us_prism * 4)
            ch_servo = 2 + i

            print(
                f"[Prismatic Servo {ch_servo}] "
                f"Extension={extension:.2f}, "
                f"us={us_prism:.1f}, "
                f"target_q={target_q_prism}"
            )
            self.maestro.set_target(ch_servo, target_q_prism)

    def get_joint_positions(self):
        """Return the array of joint positions in 3D."""
        return self.Jw.copy()

###############################################################################
# Plotting Helpers
###############################################################################
_fig_gradient = None

def update_plot(gradient_norms, plt):
    """
    Plot gradient norms vs. iteration in the second subwindow (index=1).
    """
    global _fig_gradient
    if _fig_gradient is not None:
        plt.at(1).remove(_fig_gradient)

    f = plot(
        range(len(gradient_norms)),
        gradient_norms,
        title="Gradient Norm",
        xtitle="Iteration",
        ytitle="||Grad||",
        aspect=16/9,
        axes=dict(text_scale=1.2),
    ).clone2d(pos='center', size=1)
    _fig_gradient = f
    plt.at(1).add(f)

def visualize_jacobian(arm, active_jacobian=0):
    """
    Draw arrows for J_theta and J_d for one chosen 'active_jacobian' column.
    """
    arrows = vd.Assembly()
    arrows.name = "JacobianArrows"

    J_theta, J_d = arm.velocity_jacobian()
    end_pos = arm.get_joint_positions()[-1, :]

    # Blue arrow = revolve Jacobian
    j_th = J_theta[:, active_jacobian]
    arr_th = vd.Arrow(end_pos, end_pos + j_th, s=0.01, alpha=0.4).c('blue')
    arrows += arr_th

    # Red arrow = prismatic Jacobian
    j_d_ = J_d[:, active_jacobian]
    arr_d = vd.Arrow(end_pos, end_pos + j_d_, s=0.01, alpha=0.4).c('red')
    arrows += arr_d

    return arrows

def draw_arm(arm):
    """
    Build spheres and cylinders (3D) for the revolve portion of the arm.
    """
    arm_asm = vd.Assembly()
    arm_asm.name = "Assembly"
    Jw = arm.get_joint_positions()

    # first joint (sphere)
    arm_asm += vd.Sphere(pos=Jw[0], r=0.05)
    for i in range(1, arm.n + 1):
        cyl = vd.Cylinder(pos=[Jw[i-1], Jw[i]], r=0.02)
        sph = vd.Sphere(pos=Jw[i], r=0.05)
        arm_asm += cyl
        arm_asm += sph
    return arm_asm

###############################################################################
# GUI Callbacks
###############################################################################
def on_slider_angle1(widget, event, arm, plt, active_jacobian_ref):
    val = max(0, min(widget.value, np.pi/2))
    arm.X[0] = val
    arm.FK()
    arm.update_physical_arm()

    plt.at(0).remove("Assembly")
    plt.at(0).add(draw_arm(arm))

    # Arrows
    plt.at(0).remove("JacobianArrow")
    arr = visualize_jacobian(arm, active_jacobian_ref[0])
    arr.name = "JacobianArrow"
    plt.at(0).add(arr)

    degs = np.degrees(arm.X[:2])
    print("Current revolve angles (deg):", degs)
    plt.at(0).render()

def on_slider_angle2(widget, event, arm, plt, active_jacobian_ref):
    val = max(0, min(widget.value, np.pi/2))
    arm.X[1] = val
    arm.FK()
    arm.update_physical_arm()

    plt.at(0).remove("Assembly")
    plt.at(0).add(draw_arm(arm))

    plt.at(0).remove("JacobianArrow")
    arr = visualize_jacobian(arm, active_jacobian_ref[0])
    arr.name = "JacobianArrow"
    plt.at(0).add(arr)

    degs = np.degrees(arm.X[:2])
    print("Current revolve angles (deg):", degs)
    plt.at(0).render()

def on_slider_delta1(widget, event, arm, plt, active_jacobian_ref):
    val = widget.value
    arm.X[2] = val
    arm.FK()
    arm.update_physical_arm()

    plt.at(0).remove("Assembly")
    plt.at(0).add(draw_arm(arm))

    plt.at(0).remove("JacobianArrow")
    arr = visualize_jacobian(arm, active_jacobian_ref[0])
    arr.name = "JacobianArrow"
    plt.at(0).add(arr)

    degs = np.degrees(arm.X[:2])
    print(f"Current revolve angles (deg): {degs}")
    print(f"Current prismatic: {arm.X[2:]}")
    plt.at(0).render()

def on_slider_delta2(widget, event, arm, plt, active_jacobian_ref):
    val = widget.value
    arm.X[3] = val
    arm.FK()
    arm.update_physical_arm()

    plt.at(0).remove("Assembly")
    plt.at(0).add(draw_arm(arm))

    plt.at(0).remove("JacobianArrow")
    arr = visualize_jacobian(arm, active_jacobian_ref[0])
    arr.name = "JacobianArrow"
    plt.at(0).add(arr)

    degs = np.degrees(arm.X[:2])
    print(f"Current revolve angles (deg): {degs}")
    print(f"Current prismatic: {arm.X[2:]}")
    plt.at(0).render()

def on_slider_weight_angle(widget, event, arm):
    arm.WeightAngle = widget.value

def on_slider_weight_delta(widget, event, arm):
    arm.WeightDelta = widget.value

def left_button_press(evt, arm, plt, active_jacobian_ref):
    """
    Clicking in the scene sets a new IK target and runs the solver.
    Resets the arm to X=0 for angles/deltas, then solves.
    """
    picked = evt.picked3d
    if picked is None:
        return
    ik_target = [picked[0], picked[1], 0]
    plt.at(0).remove("Sphere")
    plt.at(0).remove("JacobianArrow")
    plt.at(0).add(vd.Sphere(pos=ik_target, r=0.05, c='b'))

    # Reset angles/deltas
    arm.X = np.zeros(4)

    # Run IK
    gradient_norms = arm.IK(ik_target)
    update_plot(gradient_norms, plt)

    # Update visualization
    plt.at(0).remove("Assembly")
    plt.at(0).add(draw_arm(arm))
    plt.at(0).render()
