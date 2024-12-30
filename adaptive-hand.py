# %%
import vedo as vd
vd.settings.default_backend = 'vtk'
import numpy as np
from vedo.pyplot import plot
# %%

fig = None

def Rot(angle, axis):
    axis = np.array(axis)
    axis = axis/np.linalg.norm(axis)
    I = np.eye(3)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = I + np.sin(angle)*K + (1-np.cos(angle))*np.dot(K,K)
    return R

def UpdatePlot(iterations, gradient_norms):
    global fig
    ################# plot
    if fig is not None:
        plt.at(1).remove(fig)

    fig = plot(
        iterations, gradient_norms,
        title= "Objective",
        xtitle="steps",
        ytitle="gradient value",
        aspect=16/9,     # aspect ratio x/y of plot
        axes=dict(text_scale=1.2),
        ).clone2d(pos='center', size=1.3)

    plt.at(1).add(fig)

class SimpleArm:
    def __init__(self, n=5, L=None):
        self.n = n
        self.WeightDelta = 0.0
        self.WeightAngle = 0.0
        self.angles = [0.0] * self.n
        if L is None:
            self.L = [1.0]*self.n
        else:
            self.L = L
        
        self.min_lengths = [ l for l in self.L]
        self.delta = [0.0] * self.n
        self.Jw = np.zeros((self.n+1, 3))
        self.FK()


    ###############################################
    #                                             #
    #           VELOCITY JACOBIAN                 #
    #                                             #
    ###############################################
    def velocity_jacobian(self):
        """
        input: joint angles θ = [θ₁, θ₂, ..., θₙ], joint extensions d = [d₁, d₂, ..., dₙ]
        output: Jacobian matrices J_theta, J_d
        """
        J_theta = np.zeros((3, self.n))  # Jacobian for angular velocities
        J_d = np.zeros((3, self.n))      # Jacobian for prismatic joint velocities
        
        z_axis = np.array([0, 0, 1])  # w (omega) = Rotation axis for all joints
        
        p_e = self.Jw[-1, :]
        total_angle = 0

        for i in range(self.n):
            # Position of joint i
            p_i = self.Jw[i, :]
            
            # Angular velocity component (cross product with w)
            J_theta[:, i] = np.cross(z_axis, p_e - p_i)
            
            # Linear extension component
            total_angle += self.angles[i]
            R = Rot(total_angle, z_axis) 
            J_d[:, i] = R @ np.array([1, 0, 0])  # Direction of extension
        return J_theta, J_d

    ###############################################
    #                                             #
    #         FORWARD KINEMATICS (FK)             #
    #                                             #
    ###############################################
    def FK(self, angles=None, delta=None):
        """
        input: joint angles θ = [θ₁, θ₂, ..., θₙ], joint extensions d = [d₁, d₂, ..., dₙ]
        output: end-effector position pₑ
        """
        if angles is not None:
            self.angles = angles
        if delta is not None:
            # Ensure segment lengths don't go below minimum
            self.delta = [ d for i, d in enumerate(delta)]

        self.Jw[0, :] = np.array([0.0, 0.0, 0.0])
        total_angle = 0.0
        
        for i in range(1, self.n+1):
            total_angle += self.angles[i-1]
            R = Rot(total_angle, [0, 0, 1])
            length = max(self.L[i-1] + self.delta[i-1], self.min_lengths[i-1])
            self.Jw[i, :] = self.Jw[i-1, :] + R @ np.array([length, 0, 0])

        return self.Jw[-1, :]


    ###############################################
    #                                             #
    #         INVERSE KINEMATICS (IK)             #
    #                                             #
    ###############################################
    def IK(self, target):
        max_iterations = 1000
        alpha = 0.01
        gradient_norms = []  # List to store gradient norms for each iteration
        iterations = []  # List to store iteration count

        for i in range(max_iterations):
            current_pos = self.FK()
            error = target - current_pos
            w1 = self.WeightDelta 
            w2 = self.WeightAngle
            J_theta, J_d = self.velocity_jacobian()

            #J_theta_weighted = J_theta + w2 * 2 * np.array(self.angles)  
            #J_d_weighted = J_d + w1 * 2 * np.array(self.delta) 
            J = np.hstack((J_theta, J_d))
            JTJ = J.T @ J 

            # Construct the block matrix H 8X8
            H_alpha = 2 * w2 * np.eye(J_theta.shape[1])
            H_delta = 2 * w1 * np.eye(J_d.shape[1])
            H = np.block([[H_alpha, np.zeros((H_alpha.shape[0], H_delta.shape[1]))],
                        [np.zeros((H_delta.shape[0], H_alpha.shape[1])), H_delta]])

            # Calculate the pseudo-inverse (Hessian O^-1)
            J_dag = np.linalg.pinv(JTJ + H)

            # Gradient O
            angles_extended = np.hstack([np.array(self.angles), np.zeros(4)])  # Extend angles to 1x8
            delta_extended = np.hstack([np.zeros(4), np.array(self.delta)])  # Extend delta to 1x8

            grad = J.T @ error + 2 * w2 * angles_extended + 2 * w1 * delta_extended

            # Hessian O^-1 * grad O
            updates = alpha * (J_dag @ grad)

            # Store iteration count and gradient norm
            iterations.append(i)
            gradient_norms.append(np.linalg.norm(grad))

            # Split updates into angles and deltas
            angle_updates = updates[:self.n]
            delta_updates = updates[self.n:]

            self.angles = [angle + update for angle, update in zip(self.angles, angle_updates)]
            self.delta = [max(self.delta[i], self.delta[i] + delta_updates[i]) for i in range(self.n)]

            if np.linalg.norm(grad) < 1e-5:
                print(f"Converged in {i} iterations")
                break
        
        # Plot the gradient norm vs iterations when the loop exits
        UpdatePlot(iterations, gradient_norms)

        self.FK()






    def visualizeJacobian(self):
        J_theta, J_d = self.velocity_jacobian()  # Get both Jacobians
        # Combine them horizontally for visualization
        J = np.hstack((J_theta, J_d))
        
        max_norm = np.max(np.linalg.norm(J, axis=0))
        if max_norm == 0:
            max_norm = 1  
        total_length = sum(self.L)
        scale = 0.2 * total_length / max_norm  
        
        arrows = vd.Assembly()
        arrows.name = "JacobianArrows"
        color_list = ['red', 'green', 'blue', 'red', 'purple', 'cyan', 'magenta', 'black'][:self.n]
        
        # Visualize angular velocity components (J_theta)
        for i in range(1, self.n):
            o_i = self.Jw[i, :]  # Use joint i directly
            J_col = J_theta[:, i] * scale
            arrow = vd.Arrow(o_i, o_i + J_col, c=color_list[i % len(color_list)])
            arrows += arrow

        # Visualize linear velocity components (J_d)
        for i in range(1, self.n):
            o_i = self.Jw[i, :]  # Use joint i directly
            J_col = J_d[:, i] * scale
            arrow = vd.Arrow(o_i, o_i + 3 * J_col, c=color_list[i % len(color_list)], alpha=0.5)
            arrows += arrow

        # Add an additional vector for n+1 (end-effector position)
        o_n = self.Jw[self.n, :]  # Position of end-effector
        J_col_theta_end = J_theta[:, self.n - 1] * scale  # Angular velocity component at the end
        J_col_d_end = J_d[:, self.n - 1] * scale  # Linear velocity component at the end
        
        # Add angular velocity arrow at the end-effector
        arrow_theta_end = vd.Arrow(o_n, o_n + J_col_theta_end, c='black', alpha=0.8)
        arrows += arrow_theta_end

        # Add linear velocity arrow at the end-effector
        arrow_d_end = vd.Arrow(o_n, o_n + 3 * J_col_d_end, c='black', alpha=0.5)
        arrows += arrow_d_end

        return arrows

    def draw(self):
        vd_arm = vd.Assembly()
        vd_arm += vd.Sphere(pos=self.Jw[0, :], r=0.05)
        for i in range(1, self.n+1):
            vd_arm += vd.Cylinder(pos=[self.Jw[i-1, :], self.Jw[i, :]], r=0.02)
            vd_arm += vd.Sphere(pos=self.Jw[i, :], r=0.05)
        vd_arm+= self#.visualizeJacobian()
        return vd_arm




# %%
# Global variables and visualization setup
arm = SimpleArm(n=4, L=[1, 1, 1, 1])
IK_target = [1, 1, 0]
activeJoint = 0

def LeftButtonPress(evt):
    global IK_target
    IK_target = evt.picked3d
    plt.at(0).remove("Sphere")
    plt.at(0).add(vd.Sphere(pos=IK_target, r=0.05, c='b'))
    
    # Fully reset all delta and angle values to ensure no previous computation persists
    arm.delta = [0.0] * arm.n  # Resetting delta values
    arm.angles = [0.0] * arm.n  # Resetting angle values
    
    # Call IK with a fresh state
    arm.IK(IK_target)
    
    # Clear previous visualization and redraw the arm's state
    plt.at(0).remove("Assembly")
    plt.at(0).add(arm.draw())
    plt.at(0).render()

def OnWeightAngle(widget, event):
    arm.WeightAngle = widget.value

def OnWeightDelta(widget, event):
    arm.WeightDelta = widget.value

plt = vd.Plotter(N=2)
plt.at(0).add(arm.draw())

plt.at(0).add(vd.Sphere(pos=IK_target, r=0.05, c='b').draggable(True))  
plt.at(0).add(vd.Plane(s=[3*sum(arm.L), 3*sum(arm.L)]))

sliderWeightDelta = plt.at(0).add_slider(OnWeightDelta, 0, 1, 0.0, title="Weight Delta",title_size=0.5, pos=3, c="dg")
sliderWeightAngle = plt.at(0).add_slider(OnWeightAngle, 0, 1, 0.0, title="Weight Angle",title_size=0.5, pos=1,c="dr")

plt.at(0).add_callback('LeftButtonPress', LeftButtonPress)

plt.user_mode('2d')
plt.show(zoom="tightest" ,interactive=True)
plt.close()
# %%

