import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def is_point_in_gate(p1, p2, thickness, point):
    # Convert the points to numpy arrays for easier vector manipulation
    p1 = np.array(p1)
    p2 = np.array(p2)
    point = np.array(point)

    # Compute the direction vector of the gate segment and its length
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length == 0:
        raise ValueError("The points p1 and p2 cannot be the same.")

    # Normalize the direction vector
    direction_normalized = direction / length

    # Compute the normal vector to the direction
    normal = np.array([-direction_normalized[1], direction_normalized[0]])

    # Compute the vectors from p1 to the point and from p2 to the point
    vector_p1_to_point = point - p1

    # Project the point onto the direction vector to see if it is between p1 and p2
    projection_length = np.dot(vector_p1_to_point, direction_normalized)
    
    # Check if the point projection lies between p1 and p2
    if projection_length < 0 or projection_length > length:
        return False

    # Compute the distance of the point from the line segment
    distance_from_line = abs(np.dot(vector_p1_to_point, normal))

    # Check if the distance is within the given thickness
    if distance_from_line <= thickness:
        return True
    else:
        return False

def visualize_gate(p1, p2, thickness):
    # Initial point to visualize
    initial_point = [2, 0.5]

    # Create the plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)
    ax.set_aspect('equal')
    ax.set_xlim(-5, 10)
    ax.set_ylim(-5, 5)

    # Plot the gate segment and the boundary lines
    p1 = np.array(p1)
    p2 = np.array(p2)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    direction_normalized = direction / length
    normal = np.array([-direction_normalized[1], direction_normalized[0]])

    boundary1 = p1 + thickness * normal
    boundary2 = p2 + thickness * normal
    boundary3 = p1 - thickness * normal
    boundary4 = p2 - thickness * normal

    gate_line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2, label='Gate')
    boundary_plus, = ax.plot([boundary1[0], boundary2[0]], [boundary1[1], boundary2[1]], 'r--', label='Boundary +')
    boundary_minus, = ax.plot([boundary3[0], boundary4[0]], [boundary3[1], boundary4[1]], 'r--', label='Boundary -')

    # Plot the point
    point_plot, = ax.plot([initial_point[0]], [initial_point[1]], 'bo', label='Point')

    # Add sliders for point coordinates
    axcolor = 'lightgoldenrodyellow'
    axpointx = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axpointy = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    spointx = Slider(axpointx, 'Point X', -5.0, 10.0, valinit=initial_point[0])
    spointy = Slider(axpointy, 'Point Y', -5.0, 5.0, valinit=initial_point[1])

    # Add sliders for gate endpoints and thickness
    axp1x = plt.axes([0.25, 0.25, 0.3, 0.03], facecolor=axcolor)
    axp1y = plt.axes([0.25, 0.3, 0.3, 0.03], facecolor=axcolor)
    axp2x = plt.axes([0.65, 0.25, 0.3, 0.03], facecolor=axcolor)
    axp2y = plt.axes([0.65, 0.3, 0.3, 0.03], facecolor=axcolor)
    axthickness = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    sp1x = Slider(axp1x, 'P1 X', -5.0, 10.0, valinit=p1[0])
    sp1y = Slider(axp1y, 'P1 Y', -5.0, 5.0, valinit=p1[1])
    sp2x = Slider(axp2x, 'P2 X', -5.0, 10.0, valinit=p2[0])
    sp2y = Slider(axp2y, 'P2 Y', -5.0, 5.0, valinit=p2[1])
    sthickness = Slider(axthickness, 'Thickness', 0.1, 5.0, valinit=thickness)

    # Update function for sliders
    def update(val):
        # Update gate points and thickness
        p1_updated = np.array([sp1x.val, sp1y.val])
        p2_updated = np.array([sp2x.val, sp2y.val])
        thickness_updated = sthickness.val

        # Update gate and boundaries
        direction = p2_updated - p1_updated
        length = np.linalg.norm(direction)
        if length != 0:
            direction_normalized = direction / length
            normal = np.array([-direction_normalized[1], direction_normalized[0]])

            boundary1 = p1_updated + thickness_updated * normal
            boundary2 = p2_updated + thickness_updated * normal
            boundary3 = p1_updated - thickness_updated * normal
            boundary4 = p2_updated - thickness_updated * normal

            gate_line.set_xdata([p1_updated[0], p2_updated[0]])
            gate_line.set_ydata([p1_updated[1], p2_updated[1]])
            boundary_plus.set_xdata([boundary1[0], boundary2[0]])
            boundary_plus.set_ydata([boundary1[1], boundary2[1]])
            boundary_minus.set_xdata([boundary3[0], boundary4[0]])
            boundary_minus.set_ydata([boundary3[1], boundary4[1]])

        # Update point position
        point = [spointx.val, spointy.val]
        point_plot.set_xdata([point[0]])
        point_plot.set_ydata([point[1]])

        # Check if the point passes through the gate
        if is_point_in_gate(p1_updated, p2_updated, thickness_updated, point):
            point_plot.set_color('g')  # Green if point passes through the gate
        else:
            point_plot.set_color('b')  # Blue otherwise

        fig.canvas.draw_idle()

    spointx.on_changed(update)
    spointy.on_changed(update)
    sp1x.on_changed(update)
    sp1y.on_changed(update)
    sp2x.on_changed(update)
    sp2y.on_changed(update)
    sthickness.on_changed(update)

    # Initial check for point position
    update(None)

    # Show the plot
    plt.legend()
    plt.show()

# Example usage
p1 = (0, 0)
p2 = (4, 0)
thickness = 1
visualize_gate(p1, p2, thickness)
