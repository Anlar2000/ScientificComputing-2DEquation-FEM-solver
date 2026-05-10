import numpy as np
import scipy
# Initialize Global Matrices (using scipy.sparse for memory efficiency)
from scipy.sparse import lil_matrix
from code.FEM.Mesh.StructuredGrid import GenerateMesh


def calculate_area(coords):
    # Shoelace formula for triangle area
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * abs(x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1]))

def calculate_gradients(coords, area):
    x, y = coords[:, 0], coords[:, 1]
    # These are the constant gradients of the 3 hat functions over the triangle
    # Following the formula: grad(phi_i) = 1/(2A) * [y_j - y_k, x_k - x_j]
    dn1 = np.array([y[1]-y[2], x[2]-x[1]]) / (2*area)
    dn2 = np.array([y[2]-y[0], x[0]-x[2]]) / (2*area)
    dn3 = np.array([y[0]-y[1], x[1]-x[0]]) / (2*area)
    return [dn1, dn2, dn3]

def compute_local_stiffness(area, grads):
    K_local = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # This is the discrete version of: integral(grad(phi_i) . grad(phi_j))
            K_local[i, j] = area * np.dot(grads[i], grads[j])
    return K_local

def compute_local_mass(area):
    # This is the integral(phi_i * phi_j) for linear triangles
    # It always results in this specific 3x3 pattern:
    return (area / 12.0) * np.array([[2, 1, 1],
                                     [1, 2, 1],
                                     [1, 1, 2]])

def Assemble_K_and_M(elements,nodes):
    num_nodes = len(nodes) # Derive N total nodes
    K_global = lil_matrix((num_nodes, num_nodes))
    M_global = lil_matrix((num_nodes, num_nodes))

    for element in elements:
        # 1. Get the indices of the three nodes
        node_indices = element # e.g., [0, 1, 11]
        
        # 2. Get the actual (x, y) coordinates for these nodes
        coords = nodes[node_indices] # a 3x2 array
        
        # --- THIS IS WHERE THE AREA CODE GOES ---
        area = calculate_area(coords)
        grads = calculate_gradients(coords, area)
        
        # 3. Calculate local 3x3 matrices
        K_local = compute_local_stiffness(area, grads)
        M_local = compute_local_mass(area)
        
        # 4. "Stamp" them into the global matrices
        for i in range(3):
            for j in range(3):
                K_global[node_indices[i], node_indices[j]] += K_local[i, j]
                M_global[node_indices[i], node_indices[j]] += M_local[i, j]

    # After the loop, before the return:
    return K_global.tocsr(), M_global.tocsr()





K, M = Assemble_K_and_M(elements, nodes)
import matplotlib.pyplot as plt
plt.spy(K, markersize=1)
plt.title("Sparsity Pattern of the Global Stiffness Matrix")
plt.show()