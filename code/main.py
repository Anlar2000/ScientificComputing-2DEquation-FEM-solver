import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

# 1. Find the project root (3 levels up from Assembly)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 2. Add it to the system path so Python can find 'code'
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# 1. Mesh Generation
from FEM.Mesh.StructuredGrid import generate_mesh

# 2. Matrix Assembly
from FEM.Assembly.HeatEquationMatrixAssembler import Assemble_K_and_M

# 3. Boundary Conditions
from FEM.Boundary.DirichletBoundary import apply_boundary_conditions

# 4. Time Stepping (Assuming you created this file from the previous step)
from FEM.Solver.BackwardEuler import solve_heat_equation

#5. 3D plotting
from FEM.PostProcessing.Plotting import plot_3d_temperature
from FEM.PostProcessing.Plotting import update_3d_plot

def main():
    # --- 1. SETUP ---
    N = 25  # 10x10 grid (100 nodes total)
    dt = 0.0001
    num_steps = 100
    
    print("Generating Mesh...")
    nodes, elements = generate_mesh(N)
    
    # --- 2. ASSEMBLY ---
    print("Assembling Global Matrices...")
    K, M = Assemble_K_and_M(elements, nodes)
    
    # --- 3. BOUNDARY CONDITIONS ---
    print("Applying Dirichlet Boundary Conditions...")
    K, M = apply_boundary_conditions(K, M, nodes)
    
    # --- 4. INITIAL CONDITIONS ---
    print("Setting Initial Conditions...")
    num_nodes = len(nodes)
    U_initial = np.zeros(num_nodes)
    
    # Let's drop a "heat bomb" right in the middle of the mesh
    # Finding the node closest to (0.5, 0.5)
    distances = np.sqrt((nodes[:,0] - 0.5)**2 + (nodes[:,1] - 0.5)**2)
    # Instead of one node, make a 2x2 area hot
    # Bomb 1: Bottom-Left
    dist1 = np.sqrt((nodes[:,0] - 0.3)**2 + (nodes[:,1] - 0.3)**2)
    U_initial[dist1 < 0.12] = 100.0

    # Bomb 2: Top-Right
    dist2 = np.sqrt((nodes[:,0] - 0.7)**2 + (nodes[:,1] - 0.7)**2)
    U_initial[dist2 < 0.12] = 80.0 # Make it slightly cooler for visual contrast
    
    # --- 5. SOLVE WITH LIVE ANIMATION ---
    print("Solving and Animating...")
    plt.ion() # Turn on interactive mode
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    U_current = U_initial.copy()
    LHS = (M + dt * K).tocsr()

    for n in range(num_steps):
        # Solve step
        RHS = M.dot(U_current)
        U_next = spsolve(LHS, RHS)
        U_current = U_next

        # Update Plot every 2 steps to save CPU
        if n % 2 == 0:
            update_3d_plot(ax, nodes, elements, U_current, n)
            plt.pause(0.01) # Small pause to allow the GUI to refresh

    plt.ioff() # Turn off interactive mode
    plt.show() # Keep the final frame open
    
    # --- 6. VISUALIZE RESULT ---
    """print("Plotting Final State...")
    U_final = U_history[-1]
    
    plt.figure(figsize=(8, 6))
    plt.tripcolor(nodes[:,0], nodes[:,1], elements, U_final, shading='gouraud', cmap='inferno')
    plt.colorbar(label='Temperature')
    plt.title(f"2D Heat Equation - Time Step {num_steps}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    plot_3d_temperature(nodes,elements, U_final)"""

if __name__ == "__main__":
    main()