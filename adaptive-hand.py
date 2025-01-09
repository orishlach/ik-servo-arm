# %%
import vedo as vd
vd.settings.default_backend = 'vtk'
import numpy as np
from vedo.pyplot import plot

fig = None  # For gradient norm plotting

###############################################################################
#                           Simple Backtracking Line Search                   #
###############################################################################
def line_search(func, X, d):
    """
    Performs a simple backtracking line search:
    - If func(X + alpha*d) > func(X), reduce alpha by half.
    - Returns the final alpha.
    """
    alpha = 1.0
    while func(X + alpha*d) > func(X):
        alpha *= 0.5
        if alpha < 1e-6:
            break
    return alpha

###############################################################################
#                               Rotation Function                              #
###############################################################################
def Rot(angle, axis):
    """
    Returns a 3x3 rotation matrix for a rotation of 'angle' radians about 'axis'.
    """
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    I = np.eye(3)
    K = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],  0,         -axis[0]],
        [-axis[1], axis[0],   0      ]
    ])
    R = I + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
    return R

###############################################################################
#                    Helper for Updating the Gradient-Norm Plot               #
###############################################################################
def UpdatePlot(gradient_norms):
    global fig
    if fig is not None:
        plt.at(1).remove(fig)

    fig = plot(
        range(len(gradient_norms)), 
        gradient_norms,
        title="Gradient Norm",
        xtitle="Iteration",
        ytitle="||Grad||",
        aspect=16/9,
        axes=dict(text_scale=1.2),
    ).clone2d(pos='center', size=1)

    plt.at(1).add(fig)

###############################################################################
#                              SimpleArm Class                                #
###############################################################################
class SimpleArm:
    def __init__(self, n=5, L=None):
        """
        A simple robotic arm with n joints. We keep a single vector X of length 2*n:
           X[:n]   => the angles (revolute joints)
           X[n:]   => the deltas (prismatic extensions)

        We also define:
          - WeightDelta (w1)
          - WeightAngle (w2)
        """
        self.n = n
        self.WeightDelta = 0.0  # w1
        self.WeightAngle = 0.0  # w2

        # X = [theta_1, ..., theta_n,  delta_1, ..., delta_n]
        self.X = np.zeros(2*n)  # all angles/deltas start at 0

        # Link lengths
        if L is None:
            self.L = [1.0]*self.n
        else:
            self.L = L

        # Joint world positions (for visualization)
        self.Jw = np.zeros((self.n+1, 3))

        # Initial forward kinematics
        self.FK()

    ############################################################################
    #                               Forward Kinematics (FK)                    #
    ############################################################################
    def FK(self, X=None):
        """
        Updates self.X if X is provided, then computes the positions of all joints.
        Returns end-effector position.
        """
        if X is not None:
            self.X = X

        # angles = self.X[:n], delta = self.X[n:]
        angles = self.X[:self.n]
        delta = self.X[self.n:]

        self.Jw[0, :] = np.array([0.0, 0.0, 0.0])
        total_angle = 0.0
        for i in range(1, self.n + 1):
            total_angle += angles[i - 1]
            R = Rot(total_angle, [0, 0, 1])
            length = self.L[i - 1] + delta[i - 1]
            self.Jw[i, :] = self.Jw[i - 1, :] + R @ np.array([length, 0, 0])

        return self.Jw[-1, :]

    ############################################################################
    #                             Velocity Jacobian                             #
    ############################################################################
    def velocity_jacobian(self, X=None):
        """
        Returns (J_theta, J_d), each is 3xn, for the current state of X.
        - J_theta w.r.t. angles
        - J_d     w.r.t. deltas
        """
        if X is not None:
            self.X = X

        angles = self.X[:self.n]
        delta = self.X[self.n:]

        J_theta = np.zeros((3, self.n))
        J_d = np.zeros((3, self.n))

        z_axis = np.array([0, 0, 1])
        p_end = self.Jw[-1, :]
        total_angle = 0.0

        for i in range(self.n):
            p_i = self.Jw[i, :]
            # angular part
            J_theta[:, i] = np.cross(z_axis, p_end - p_i)

            # prismatic part
            total_angle += angles[i]
            R = Rot(total_angle, z_axis)
            J_d[:, i] = R @ np.array([1, 0, 0])

        return J_theta, J_d

    ############################################################################
    #                          Inverse Kinematics (IK)                          #
    ############################################################################
    def IK(self, target):
        """
        Uses a Newton-like approach with pseudo-inverse of (J^T J + H)
        and performs a line search with the objective function.

        The objective is: O = w2*||angles||^2 + w1*||delta||^2 + ||error||^2

        Where angles = X[:n], delta = X[n:].
        """
        max_iterations = 1000
        w1 = self.WeightDelta  # Weight for deltas
        w2 = self.WeightAngle  # Weight for angles

        def objective_value(vecX):
            """
            vecX is [angles..., delta...], length=2n.
            We'll temporarily set self.X to vecX, do an FK, compute error, etc.
            Then restore old X after.
            """
            oldX = self.X.copy()
            self.FK(vecX)  # sets self.X internally
            current_pos = self.Jw[-1, :]
            error_vec = current_pos - target

            # angles, deltas:
            a = vecX[:self.n]
            d = vecX[self.n:]

            norm_angle_sqr = np.sum(a**2)
            norm_delta_sqr = np.sum(d**2)
            norm_error_sqr = np.sum(error_vec**2)

            cost = w2*norm_angle_sqr + w1*norm_delta_sqr + norm_error_sqr

            # restore old X and old Jw
            self.FK(oldX)
            return cost

        gradient_norms = []

        for i in range(max_iterations):
            # Current end-effector / error
            current_pos = self.FK()  # uses self.X
            error = current_pos - target

            # Angles & deltas from self.X
            angles = self.X[:self.n]
            delta = self.X[self.n:]

            # Build Jacobian
            J_theta, J_d = self.velocity_jacobian()
            J = np.hstack((J_theta, J_d))  # shape (3, 2n)

            # Extended vectors for gradient
            a_extended = np.hstack([angles, np.zeros(self.n)])  # angles in first n
            d_extended = np.hstack([np.zeros(self.n), delta])    # deltas in last n

            # gradient of O
            grad = J.T @ error + 2*w2*a_extended + 2*w1*d_extended

            # build Hessian approx = (J^T J + H)
            H_a = 2*w2*np.eye(self.n)
            H_d = 2*w1*np.eye(self.n)
            H = np.block([
                [H_a, np.zeros((self.n, self.n))],
                [np.zeros((self.n, self.n)), H_d]
            ])

            JTJ = J.T @ J
            A = JTJ + H
            A_inv = np.linalg.pinv(A)

            # direction
            dX = -(A_inv @ grad)

            # do a line search
            x_current = self.X.copy()
            alpha = line_search(objective_value, x_current, dX)
            step = alpha * dX

            # update self.X
            self.X += step

            # track gradient norm
            grad_norm = np.linalg.norm(grad)
            gradient_norms.append(grad_norm)

            if grad_norm < 1e-6:
                print(f"Converged in {i} iterations")
                break

        # finalize plotting of the gradient norm
        UpdatePlot(gradient_norms)
        print(f"Angle: {angles}")
        # final forward kinematics update
        self.FK()

    ############################################################################
    #                          Visualizing Jacobian Arrows                      #
    ############################################################################
    def visualizeJacobian(self):
        """
        Draws two arrows: one for J_theta[:, activeJacobian], one for J_d[:, activeJacobian].
        Uses the end-effector as the arrow origin.
        """
        global activeJacobian
        arrows = vd.Assembly()
        arrows.name = "JacobianArrows"

        J_theta, J_d = self.velocity_jacobian()
        end_pos = self.Jw[-1, :]

        j_theta_col = J_theta[:, activeJacobian]
        arrow_theta = vd.Arrow(end_pos, end_pos + j_theta_col, s=0.01, c='blue', alpha=0.4)
        arrows += arrow_theta

        j_d_col = J_d[:, activeJacobian]
        arrow_delta = vd.Arrow(end_pos, end_pos + j_d_col, s=0.01, c='red', alpha=0.4)
        arrows += arrow_delta

        return arrows

    ############################################################################
    #                           Drawing the Arm Assembly                        #
    ############################################################################
    def draw(self):
        """
        Returns a vedo.Assembly with spheres for joints and cylinders for links.
        """
        vd_arm = vd.Assembly()
        vd_arm += vd.Sphere(pos=self.Jw[0, :], r=0.05)
        for i in range(1, self.n+1):
            c = vd.Cylinder(pos=[self.Jw[i-1, :], self.Jw[i, :]], r=0.02)
            s = vd.Sphere(pos=self.Jw[i, :], r=0.05)
            vd_arm += c
            vd_arm += s
        vd_arm += self
        return vd_arm

    def show_arrows(self):
        """
        Removes old "JacobianArrow" if present, and adds the new arrow set.
        """
        plt.at(0).remove("JacobianArrow")
        arr = self.visualizeJacobian()
        arr.name = "JacobianArrow"
        plt.at(0).add(arr)

# -----------------------------------------------------------------------------
# Global variables and visualization
# -----------------------------------------------------------------------------
arm = SimpleArm(n=4, L=[1, 1, 1, 1])
IK_target = [1, 1, 0]
activeJoint = 0
activeJacobian = 0

# Vedo plotter with 2 subwindows
plt = vd.Plotter(N=2)

# Add the arm to subplot 0
plt.at(0).add(arm.draw())

# Add a sphere for the target
plt.at(0).add(vd.Sphere(pos=IK_target, r=0.05, c='b').draggable(True))
plt.at(0).add(vd.Plane(s=[3*sum(arm.L), 3*sum(arm.L)]))

###############################################################################
#                               Slider Callbacks                              #
###############################################################################
def OnSliderAngle(widget, event):
    """
    Updates the angle of the currently active joint 
    in the first half of arm.X (angles).
    """
    global activeJoint
    arm.X[activeJoint] = widget.value  # angles are X[:n]
    arm.FK()
    arm.velocity_jacobian()
    arm.show_arrows()
    plt.at(0).remove("Assembly")
    plt.at(0).add(arm.draw())
    plt.at(0).render()

def OnSliderDelta(widget, event):
    """
    Updates the prismatic extension for the currently active joint
    in the second half of arm.X (delta).
    """
    global activeJoint
    n = arm.n
    arm.X[n + activeJoint] = widget.value  # deltas are X[n:]
    arm.FK()
    arm.velocity_jacobian()
    arm.show_arrows()
    plt.at(0).remove("Assembly")
    plt.at(0).add(arm.draw())
    plt.at(0).render()

def OnCurrentJoint(widget, event):
    """
    Changes which joint index is active for angle/delta manipulation.
    Also updates the sliderAngle to the current angle for that joint.
    """
    global activeJoint
    activeJoint = round(widget.value)
    sliderAngle.value = arm.X[activeJoint]

def OnCurrentJacobian(widget, event):
    """
    Selects which column of J to visualize for Jacobian arrows.
    """
    global activeJacobian
    activeJacobian = round(widget.value)
   # print("Active Jacobian:", activeJacobian)

def LeftButtonPress(evt):
    """
    On left-click, pick a new target in 2D (force z=0),
    reset arm.X to all zeros, run IK, redraw.
    """
    global IK_target
    picked = evt.picked3d
    if picked is None:
        return
    IK_target = [picked[0], picked[1], 0.0]
    #print("IK Target:", IK_target)

    plt.at(0).remove("Sphere")
    plt.at(0).remove("JacobianArrow")
    plt.at(0).add(vd.Sphere(pos=IK_target, r=0.05, c='b'))

    # Reset X to zeros
    arm.X = np.zeros(2*arm.n)

    # Run IK
    arm.IK(IK_target)

    # Redraw
    plt.at(0).remove("Assembly")
    plt.at(0).add(arm.draw())
    plt.at(0).render()

def OnWeightAngle(widget, event):
    arm.WeightAngle = widget.value

def OnWeightDelta(widget, event):
    arm.WeightDelta = widget.value

###############################################################################
#                         Creating the Interface                              #
###############################################################################
sliderCurrentJoint = plt.at(0).add_slider(
    OnCurrentJoint, 0, arm.n-1, 0,
    title="Current joint", title_size=0.7,
    pos=[(0.03, 0.06), (0.15, 0.06)], delayed=True
)
sliderAngle = plt.at(0).add_slider(
    OnSliderAngle, -np.pi, np.pi, 0.,
    title="Joint Angle", title_size=0.7,
    pos=[(0.20, 0.06), (0.32, 0.06)]
)
sliderCurrentJacobian = plt.at(0).add_slider(
    OnCurrentJacobian, 0, arm.n-1, 0,
    title="Joint Arrow", title_size=0.7,
    pos=[(0.37, 0.92), (0.49, 0.92)], delayed=True
)
sliderDelta = plt.add_slider(
    OnSliderDelta, -0.5, 0.5, arm.X[arm.n + activeJoint],
    title="Delta Length", title_size=0.7,
    pos=[(0.37, 0.06), (0.49, 0.06)]
)
sliderWeightDelta = plt.at(0).add_slider(
    OnWeightDelta, 0, 1, 0.0,
    title="W_Delta", title_size=0.7,
    pos=[(0.05, 0.92), (0.15, 0.92)],
    c="dr"
)
sliderWeightAngle = plt.at(0).add_slider(
    OnWeightAngle, 0, 1, 0.0,
    title="W_Angle", title_size=0.7,
    pos=[(0.20, 0.92), (0.30, 0.92)],
    c="dr"
)

plt.at(0).add_callback('LeftButtonPress', LeftButtonPress)
plt.user_mode('2d')
plt.show(zoom="tightest", interactive=True)
plt.close()
