import time
import serial  # PySerial for communication with the Maestro
import numpy as np
import vedo as vd
from vedo.pyplot import plot

###############################################################################
# Pololu Maestro helper class
###############################################################################
class PololuMaestro:
    """
    Simple interface for sending commands to the Pololu Maestro
    using the compact protocol over a serial port.
    """
    def __init__(self, port="COM3", baudrate=9600):
        """
        - port: e.g. "COM3" on Windows, or "/dev/ttyACM0" on Linux/Mac.
        - baudrate: must match your Maestro configuration 
                    (often 9600, 57600, or 115200).
        """
        self.ser = None
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
            # Wait a moment for the device to be ready
            time.sleep(2)
            print(f"Connected to Pololu Maestro on {port}.")
        except Exception as e:
            print(f"Warning: Could not open Maestro on {port}. No hardware connected.\n{e}")

    def set_target(self, channel, target):
        """
        Sets the servo target for 'channel' in quarter-microseconds.
          Example: 1500 µs -> 6000, 500 µs -> 2000, 2500 µs -> 10000, etc.
        - channel: integer (0..5 on a 6-channel Maestro).
        - target: integer in [2000..10000] for 500..2500 µs (multiplied by 4).
        """
        if self.ser is None or not self.ser.is_open:
            # If there's no open serial connection, do nothing
            return

        # Compact protocol command: 0x84, channel, lowbits(target), highbits(target)
        cmd = bytearray([0x84, channel, target & 0x7F, (target >> 7) & 0x7F])
        self.ser.write(cmd)

    def close(self):
        """ Close the serial port when done. """
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

###############################################################################
# Simple backtracking line search (for IK)
###############################################################################
def line_search(func, X, d, max_ls_iters=10):
    alpha = 1.0
    iteration = 0
    while func(X + alpha*d) > func(X):
        alpha *= 0.5
        iteration += 1
        if iteration >= max_ls_iters:
            break
        if alpha < 1e-6:
            break
    return alpha

###############################################################################
# Rotation function
###############################################################################
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
# For live plotting of gradient norms
###############################################################################
fig = None
msg = vd.Text2D(pos='bottom-left', font="VictorMono")  # text object to display final angles

def UpdatePlot(gradient_norms):
    global fig
    if fig is not None:
        plt.at(1).remove(fig)
    f = plot(
        range(len(gradient_norms)),
        gradient_norms,
        title="Gradient Norm",
        xtitle="Iteration",
        ytitle="||Grad||",
        aspect=16/9,
        axes=dict(text_scale=1.2),
    ).clone2d(pos='center', size=1)
    fig = f
    plt.at(1).add(fig)

###############################################################################
# SimpleArm class
###############################################################################
class SimpleArm:
    """
    This arm has 2 revolute joints and 2 prismatic joints:
      X = [theta1, theta2, delta1, delta2].
      So X has length 4 in total.

    Indices:
      - X[0], X[1]: angles in radians
      - X[2], X[3]: prismatic extension in [0..4.5] (example)

    Also includes a PololuMaestro object to move real servos whenever angles change.

    We do NOT call update_physical_arm() inside FK() to avoid hammering the hardware.
    Instead, we call update_physical_arm() either from sliders or after IK.
    """
    def __init__(self, maestro_port="COM3", maestro_baud=9600):
        # 2 revolve, 2 prismatic => total of 4
        self.n = 2  # number of revolute joints
        # Weights for the IK objective (managed by user sliders)
        self.WeightDelta = 0.0
        self.WeightAngle = 0.0

        # State vector: angles in X[:2], prismatic deltas in X[2:].
        self.X = np.zeros(4)

        # For GUI visualization only: each link = 14.0
        self.L = [14.0, 14.0]  # for 2 revolve links

        # Joint world positions (3D)
        self.Jw = np.zeros((self.n + 1, 3))

        # Connect to the Maestro. If it fails, self.maestro.ser will be None.
        self.maestro = PololuMaestro(port=maestro_port, baudrate=maestro_baud)

        # Initial forward kinematics
        self.FK()

    def FK(self, X=None):
        """
        Forward Kinematics: compute 3D positions of the revolve joints.
        NOTE: The prismatic deltas do not affect these link lengths in a direct
        2D sense here, but we show how you might incorporate them if you want.
        For example, each revolve link can be extended by delta[i].
        """
        if X is not None:
            self.X = X

        angles = self.X[:2]   # revolve angles, in radians
        deltas = self.X[2:]   # prismatic extensions
        self.Jw[0, :] = [0, 0, 0]
        total_angle = 0.0

        # For each revolve link:
        for i in range(1, self.n + 1):
            total_angle += angles[i - 1]
            R = Rot(total_angle, [0, 0, 1])
            # example: link i-1 length base + some prismatic extension
            # clamp link to 4.5 max for demonstration
            length_i = min(self.L[i - 1] + deltas[i - 1], 4.5)
            self.Jw[i, :] = self.Jw[i - 1, :] + R @ np.array([length_i, 0, 0])

        return self.Jw[-1, :]

    def update_physical_arm(self):
        """
        Send commands to 4 Maestro channels:
          - Channels 0..1: revolve joints (angles)
          - Channels 2..3: prismatic joints (gear extension)

        We print out the values being sent so you can see them in the console.
        """
        if not self.maestro or not self.maestro.ser or not self.maestro.ser.is_open:
            return

        # 2 revolve angles in radians
        revolve_angles_rad = self.X[:2]
        revolve_angles_rad[1] += np.radians(90)
        # 2 prismatic extensions (in [0..4.5])
        prismatic = self.X[2:]

        # -----------------------------------------------------
        # Channels 0..1 => revolve angles
        # -----------------------------------------------------
        for ch in range(2):
            angle_deg = np.degrees(revolve_angles_rad[ch])
            # Expand angle from 0..90° to 0..180° physically (example)
            angle_deg_expanded = angle_deg * 1.0

            # Map 0..180° => 500..2500 µs
            us = 500 + (angle_deg_expanded / 180.0) * 2000
            target_q = int(us * 4)

            print(
                f"[Revolve Servo {ch}] "
                f"Raw angle (deg)={angle_deg:.2f}, "
                f"Expanded angle (deg)={angle_deg_expanded:.2f}, "
                f"us={us:.1f}, "
                f"target_q={target_q}"
            )

            self.maestro.set_target(ch, target_q)

        # -----------------------------------------------------
        # Channels 2..3 => prismatic gear extension
        # Example: we map extension 0..4.5 => 500..2500 µs linearly
        # -----------------------------------------------------
        for i in range(2):
            extension = prismatic[i]
            # Map extension => servo pulse
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

    def velocity_jacobian(self, X=None):
        """
        Returns J_theta, J_d (3 x 2 each), for the revolve + prismatic.
        This is only used in IK for the revolve portion; you'd adapt for prismatic
        if you want more advanced usage.
        """
        if X is not None:
            self.X = X

        angles = self.X[:2]
        deltas = self.X[2:]
        J_theta = np.zeros((3, 2))
        J_d = np.zeros((3, 2))

        z_axis = np.array([0, 0, 1])
        p_end = self.Jw[-1, :]
        total_angle = 0.0

        for i in range(2):
            p_i = self.Jw[i, :]
            # revolve partial derivative
            J_theta[:, i] = np.cross(z_axis, p_end - p_i)

            total_angle += angles[i]
            R = Rot(total_angle, z_axis)
            # prismatic partial derivative
            J_d[:, i] = R @ np.array([1, 0, 0])

        return J_theta, J_d

    def IK(self, target):
        """
        Simple pseudo-Newton approach to minimize:
          WeightAngle*||angles||^2 + WeightDelta*||deltas||^2 + ||end-effector - target||^2
        with a line-search. Angles clamped to [0, pi], deltas to [0, 4.5].

        After it converges, we call update_physical_arm() once.
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
            cost = w2*np.sum(a**2) + w1*np.sum(d**2) + np.sum(err_vec**2)
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

            a_ext = np.hstack([angles, np.zeros(2)])
            d_ext = np.hstack([np.zeros(2), deltas])
            grad = J.T @ error + 2*w2*a_ext + 2*w1*d_ext

            # Simple Hessian approximation
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
            step = alpha * dX
            self.X += step

            # Clamp angles to [0, pi] and deltas to [0, 4.5]
            self.X[0] = np.clip(self.X[0], 0, np.pi)
            self.X[1] = np.clip(self.X[1], 0, np.pi)
            self.X[2] = np.clip(self.X[2], 0.0, 4.5)
            self.X[3] = np.clip(self.X[3], 0.0, 4.5)

            gn = np.linalg.norm(grad)
            gradient_norms.append(gn)
            if gn < 1e-6:
                print(f"Converged in {i} iterations.")
                break

        UpdatePlot(gradient_norms)
        final_deg = np.degrees(self.X[:2])  
        final_deltas = self.X[2:]
        print("Final angles (deg):", ", ".join([f"{fd:.2f}" for fd in final_deg]))
        print("Final deltas:", ", ".join([f"{fd:.2f}" for fd in final_deltas]))

        # Update servos once
        self.update_physical_arm()

    def visualizeJacobian(self):
        """
        Draw arrows for J_theta and J_d for one chosen 'activeJacobian' column (global).
        """
        global activeJacobian
        arrows = vd.Assembly()
        arrows.name = "JacobianArrows"
        J_theta, J_d = self.velocity_jacobian()
        end_pos = self.Jw[-1, :]

        j_th = J_theta[:, activeJacobian]
        arr_th = vd.Arrow(end_pos, end_pos + j_th, s=0.01, alpha=0.4).c('blue')
        arrows += arr_th

        j_d_ = J_d[:, activeJacobian]
        arr_d = vd.Arrow(end_pos, end_pos + j_d_, s=0.01, alpha=0.4).c('red')
        arrows += arr_d
        return arrows

    def draw(self):
        """
        Assembles spheres (joints) and cylinders (links) for 3D visualization.
        Only the revolve portion is drawn here. 
        """
        arm_asm = vd.Assembly()
        arm_asm += vd.Sphere(pos=self.Jw[0], r=0.05)
        for i in range(1, self.n+1):
            cyl = vd.Cylinder(pos=[self.Jw[i-1], self.Jw[i]], r=0.02)
            sph = vd.Sphere(pos=self.Jw[i], r=0.05)
            arm_asm += cyl
            arm_asm += sph
        arm_asm += self
        return arm_asm

    def show_arrows(self):
        plt.at(0).remove("JacobianArrow")
        arr = self.visualizeJacobian()
        arr.name = "JacobianArrow"
        plt.at(0).add(arr)

###############################################################################
# Main code with Plotter
###############################################################################
arm = SimpleArm(maestro_port="COM3", maestro_baud=9600)

# The target we want the end-effector to reach
IK_target = [1, 1, 0]

activeJacobian = 0

# Plotter with 2 subwindows
plt = vd.Plotter(N=2)

# Subplot 0: the robotic arm and the target sphere
plt.at(0).add(arm.draw())
plt.at(0).add(vd.Sphere(pos=IK_target, r=0.05, c='b').draggable(True))
plt.at(0).add(vd.Plane(s=[3*sum(arm.L), 3*sum(arm.L)]))

###############################################################################
# Slider callbacks
###############################################################################
def OnSliderAngle1(widget, event):
    """
    Slider for the FIRST revolve angle (channel 0).
    We'll clamp 0..(pi/2), effectively 0..90 deg, which becomes 0..180 deg physically.
    Then we call update_physical_arm() once at the end.
    """
    val = max(0, min(widget.value, np.pi/2))
    arm.X[0] = val
    arm.FK()
    arm.velocity_jacobian()
    arm.show_arrows()

    # Update revolve servo channel 0
    arm.update_physical_arm()

    degs = np.degrees(arm.X[:2])
    print("Current revolve angles (deg):", degs)
    plt.at(0).remove("Assembly")
    plt.at(0).add(arm.draw())
    plt.at(0).render()

def OnSliderAngle2(widget, event):
    """
    Slider for the SECOND revolve angle (channel 1).
    """
    val = max(0, min(widget.value, np.pi/2))
    arm.X[1] = val
    arm.FK()
    arm.velocity_jacobian()
    arm.show_arrows()

    # Update revolve servo channel 1
    arm.update_physical_arm()

    degs = np.degrees(arm.X[:2])
    print("Current revolve angles (deg):", degs)
    plt.at(0).remove("Assembly")
    plt.at(0).add(arm.draw())
    plt.at(0).render()

def OnSliderDelta1(widget, event):
    """
    Slider for the FIRST prismatic extension (channel 2).
    Range might be -0.5..0.5 in the UI, but we clamp or interpret it as 0..4.5 in IK.
    """
    val = widget.value
    arm.X[2] = val
    arm.FK()
    arm.velocity_jacobian()
    arm.show_arrows()

    # If you want immediate servo update for prismatic, do so here:
    arm.update_physical_arm()

    degs = np.degrees(arm.X[:2])
    prismatic = arm.X[2:]
    print(f"Current revolve angles (deg): {degs}")
    print(f"Current prismatic: {prismatic}")
    plt.at(0).remove("Assembly")
    plt.at(0).add(arm.draw())
    plt.at(0).render()

def OnSliderDelta2(widget, event):
    """
    Slider for the SECOND prismatic extension (channel 3).
    """
    val = widget.value
    arm.X[3] = val
    arm.FK()
    arm.velocity_jacobian()
    arm.show_arrows()

    arm.update_physical_arm()

    degs = np.degrees(arm.X[:2])
    prismatic = arm.X[2:]
    print(f"Current revolve angles (deg): {degs}")
    print(f"Current prismatic: {prismatic}")
    plt.at(0).remove("Assembly")
    plt.at(0).add(arm.draw())
    plt.at(0).render()

def LeftButtonPress(evt):
    """
    Clicking in the scene sets a new IK target and runs the solver.
    Resets the arm to X=0 for angles/deltas, then solves.
    """
    global IK_target
    picked = evt.picked3d
    if picked is None:
        return
    IK_target = [picked[0], picked[1], 0]
    plt.at(0).remove("Sphere")
    plt.at(0).remove("JacobianArrow")
    plt.at(0).add(vd.Sphere(pos=IK_target, r=0.05, c='b'))

    # Reset angles/deltas
    arm.X = np.zeros(4)
    # Run IK (updates the servo once at the end)
    arm.IK(IK_target)

    plt.at(0).remove("Assembly")
    plt.at(0).add(arm.draw())
    plt.at(0).render()

def OnWeightAngle(widget, event):
    arm.WeightAngle = widget.value

def OnWeightDelta(widget, event):
    arm.WeightDelta = widget.value

###############################################################################
# Create sliders
###############################################################################
sliderAngle1 = plt.at(1).add_slider(
    OnSliderAngle1,
    0, np.pi/2, 0,
    title="Angle Joint 1 (0..90 deg)",
    title_size=0.7,
    pos=[(0.03, 0.06), (0.15, 0.06)],
    c="db"
)

sliderAngle2 = plt.at(1).add_slider(
    OnSliderAngle2,
    0, np.pi/2, 0,
    title="Angle Joint 2 (0..90 deg)",
    title_size=0.7,
    pos=[(0.20, 0.06), (0.32, 0.06)],
    c="dg"
)

sliderDelta1 = plt.at(1).add_slider(
    OnSliderDelta1,
    -0.5, 4.5, 0,
    title="Delta Joint 1 (prismatic)",
    title_size=0.7,
    pos=[(0.03, 0.92), (0.15, 0.92)],
    c="dr"
)

sliderDelta2 = plt.at(1).add_slider(
    OnSliderDelta2,
    -0.5, 4.5, 0,
    title="Delta Joint 2 (prismatic)",
    title_size=0.7,
    pos=[(0.20, 0.92), (0.32, 0.92)],
    c="dr"
)

sliderWeightDelta = plt.at(1).add_slider(
    OnWeightDelta, 0, 1, 0,
    title="W_Delta", title_size=0.7,
    pos=[(0.37, 0.92), (0.49, 0.92)],
    c="dr"
)

sliderWeightAngle = plt.at(1).add_slider(
    OnWeightAngle, 0, 1, 0,
    title="W_Angle", title_size=0.7,
    pos=[(0.20, 0.82), (0.32, 0.82)],
    c="dr"
)

###############################################################################
# Plotter setup
###############################################################################
plt.at(0).add_callback('LeftButtonPress', LeftButtonPress)
plt.user_mode('2d')  # 2D scene interaction
plt.show(zoom="tightest", interactive=True)
plt.close()
