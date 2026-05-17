import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_3d_temperature(nodes, elements, U_vector):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # We use the x and y from nodes, and the Temperature (U) as the z-axis
    surf = ax.plot_trisurf(nodes[:, 0], nodes[:, 1], U_vector, 
                           triangles=elements, 
                           cmap='inferno', linewidth=0, antialiased=True)
    
    ax.set_title("3D Thermal Distribution")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Temperature")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()


def update_3d_plot(ax, nodes, elements, U_vector, time_step):
    ax.clear()
    surf = ax.plot_trisurf(nodes[:, 0], nodes[:, 1], U_vector, 
                           triangles=elements, 
                           cmap='inferno', linewidth=0, antialiased=True)
    
    ax.set_title(f"Step {time_step} - Peak Temp: {np.max(U_vector):.4f}")
    
    # CRITICAL: This stops the "flattening" effect
    ax.set_zlim(0, 100) 
    
    # Optional: Keep the view angle consistent while it animates
    ax.view_init(elev=30, azim=45) 
    
    return surf