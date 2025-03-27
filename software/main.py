import numpy as np
import vedo as vd

from simple_arm import (
    SimpleArm,
    draw_arm,
    visualize_jacobian,
    on_slider_angle1,
    on_slider_angle2,
    on_slider_delta1,
    on_slider_delta2,
    on_slider_weight_angle,
    on_slider_weight_delta,
    left_button_press
)

def main():
    # Create the robotic arm
    arm = SimpleArm(maestro_port="COM3", maestro_baud=9600)

    # We can choose which Jacobian column is "active" for arrow display
    active_jacobian_ref = [0]  # store in a one-element list so we can pass by reference

    # Setup a 2-subwindow Plotter
    plt = vd.Plotter(N=2)

    # Subplot 0: arm and a default sphere as target
    default_target = [1, 1, 0]
    sphere_target = vd.Sphere(pos=default_target, r=0.05, c='b').draggable(True)
    plane = vd.Plane(s=[3*sum(arm.L), 3*sum(arm.L)])
    plt.at(0).add(draw_arm(arm))
    plt.at(0).add(sphere_target)
    plt.at(0).add(plane)

    # Add callback for left-button clicks in subplot 0
    plt.at(0).add_callback('LeftButtonPress', lambda evt: left_button_press(evt, arm, plt, active_jacobian_ref))

    # Create sliders in subplot 1
    slider_angle1 = plt.at(1).add_slider(
        callback=lambda w, e: on_slider_angle1(w, e, arm, plt, active_jacobian_ref),
        xmin=0, xmax=np.pi/2, value=0,
        title="Angle Joint 1 (0..90 deg)",
        title_size=0.7,
        pos=[(0.03, 0.06), (0.15, 0.06)],
        c="db"
    )

    slider_angle2 = plt.at(1).add_slider(
        callback=lambda w, e: on_slider_angle2(w, e, arm, plt, active_jacobian_ref),
        xmin=0, xmax=np.pi/2, value=0,
        title="Angle Joint 2 (0..90 deg)",
        title_size=0.7,
        pos=[(0.20, 0.06), (0.32, 0.06)],
        c="dg"
    )

    slider_delta1 = plt.at(1).add_slider(
        callback=lambda w, e: on_slider_delta1(w, e, arm, plt, active_jacobian_ref),
        xmin=-0.5, xmax=4.5, value=0,
        title="Delta Joint 1 (prismatic)",
        title_size=0.7,
        pos=[(0.03, 0.92), (0.15, 0.92)],
        c="dr"
    )

    slider_delta2 = plt.at(1).add_slider(
        callback=lambda w, e: on_slider_delta2(w, e, arm, plt, active_jacobian_ref),
        xmin=-0.5, xmax=4.5, value=0,
        title="Delta Joint 2 (prismatic)",
        title_size=0.7,
        pos=[(0.20, 0.92), (0.32, 0.92)],
        c="dr"
    )

    slider_weight_delta = plt.at(1).add_slider(
        callback=lambda w, e: on_slider_weight_delta(w, e, arm),
        xmin=0, xmax=1, value=0,
        title="W_Delta",
        title_size=0.7,
        pos=[(0.37, 0.92), (0.49, 0.92)],
        c="dr"
    )

    slider_weight_angle = plt.at(1).add_slider(
        callback=lambda w, e: on_slider_weight_angle(w, e, arm),
        xmin=0, xmax=1, value=0,
        title="W_Angle",
        title_size=0.7,
        pos=[(0.20, 0.82), (0.32, 0.82)],
        c="dr"
    )

    # Start interactive plot
    plt.user_mode('2d')
    plt.show(zoom="tightest", interactive=True)
    plt.close()

if __name__ == "__main__":
    main()
