# %%
import vedo as vd
vd.settings.default_backend = 'vtk'
import numpy as np

def Rot(angle, axis):
    axis = np.array(axis)
    axis = axis/np.linalg.norm(axis)
    I = np.eye(3)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = I + np.sin(angle)*K + (1-np.cos(angle))*np.dot(K,K)
    return R

class SimpleArm:
    def __init__(self, n=5, L=None):
        self.n = n
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
        """
        input: target position pₑ
        """
        max_iterations = 4000
        alpha = 0.1  # Step size
        lambda_sq = 0.1  # Damping factor
        
        for _ in range(max_iterations):
            current_pos = self.FK()
            error = target - current_pos
            
            
            w1 = sliderWeightDelta.value  
            w2 = sliderWeightAngle.value  
            '''
            norm_alpha_squared = np.sum(np.array(self.angles) ** 2)  # ||alpha||^2
            norm_delta_squared = np.sum(np.array(self.delta) ** 2)  # ||delta||^2
            norm_error_squared = np.sum(error ** 2)  # ||error||^2
            penalty_delta = w1 * norm_delta_squared
            penalty_alpha = w2 * norm_alpha_squared
            total_penalty = penalty_alpha + penalty_delta + norm_error_squared
            '''
            
            if np.linalg.norm(error) < 1e-5:
                #print(f"Penalty : {total_penalty:.3f}") 
                break
                
            J_theta, J_d = self.velocity_jacobian()
            
                # Weighted Jacobian terms
            J_theta_weighted = J_theta + w2 * 2 * np.sum(np.array(self.angles))
            J_d_weighted = J_d + w1 * 2 * np.sum(np.array(self.delta))
            # Combine them
            J = np.hstack((J_theta_weighted, J_d_weighted))

            # Damped Least Squares
            J_dag = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(3))
            
            # Compute joint updates (both angles and extensions)
            updates = alpha * (J_dag @ error)
            
            # Split updates between angles and extensions
            angle_updates = updates[:self.n]
            delta_updates = updates[self.n:]
            
            # Update joint angles
            self.angles = [angle + update for angle, update in zip(self.angles, angle_updates)]
            
            # update delta values 
            self.delta = [ max(self.delta[i] , self.delta[i] + delta_updates[i]) for i in range(self.n)]
            
        self.FK()  # Final update of joint positions
    
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
        color_list = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow'][:self.n]
        
        # Visualize angular velocity components (J_theta)
        for i in range(self.n):
            o_i = self.Jw[i,:]
            J_col = J_theta[:,i] * scale
            arrow = vd.Arrow(o_i, o_i + J_col, c=color_list[i % len(color_list)])
            arrows += arrow

        
        # Visualize linear velocity components (J_d)
        for i in range(self.n):
            o_i = self.Jw[i,:]
            J_col = J_d[:,i] * scale
            arrow = vd.Arrow(o_i, o_i + J_col, c=color_list[i % len(color_list)], alpha=0.5)
            arrows += arrow

            
        return arrows

    def draw(self):
        vd_arm = vd.Assembly()
        vd_arm += vd.Sphere(pos=self.Jw[0, :], r=0.05)
        for i in range(1, self.n+1):
            vd_arm += vd.Cylinder(pos=[self.Jw[i-1, :], self.Jw[i, :]], r=0.02)
            vd_arm += vd.Sphere(pos=self.Jw[i, :], r=0.05)
        vd_arm+= self.visualizeJacobian()
        return vd_arm

# %%
# Global variables and visualization setup
arm = SimpleArm(n=4, L=[1, 1, 1, 1])
IK_target = [1, 1, 0]
activeJoint = 0

def LeftButtonPress(evt):
    global IK_target
    IK_target = evt.picked3d
    plt.remove("Sphere")
    plt.add(vd.Sphere(pos=IK_target, r=0.05, c='b'))
    
    # Fully reset all delta and angle values to ensure no previous computation persists
    arm.delta = [0.0] * arm.n  # Resetting delta values
    arm.angles = [0.0] * arm.n  # Resetting angle values
    
    # Call IK with a fresh state
    arm.IK(IK_target)
    
    # Clear previous visualization and redraw the arm's state
    plt.remove("Assembly")
    plt.add(arm.draw())
    plt.render()

def OnWeightAngle(widget, event):
    pass

def OnWeightDelta(widget, event):
    pass

plt = vd.Plotter()
plt += arm.draw()
plt += vd.Sphere(pos=IK_target, r=0.05, c='b').draggable(True)
plt += vd.Plane(s=[3*sum(arm.L), 3*sum(arm.L)])

sliderWeightDelta = plt.add_slider(OnWeightDelta, 0, 1, 0.0, title="Weight Delta",title_size=0.5, pos=[(0.7, 0.04), (0.95, 0.04)], c="dg")
sliderWeightAngle = plt.add_slider(OnWeightAngle, 0, 1, 0.0, title="Weight Angle",title_size=0.5, pos=[(0.05, 0.04), (0.30, 0.04)],c="dr")

plt.add_callback('LeftButtonPress', LeftButtonPress)

plt.user_mode('2d').show(zoom="tightest")
plt.close()
# %%