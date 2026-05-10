import sys
import os

# 1. Find the project root (3 levels up from Assembly)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 2. Add it to the system path so Python can find 'code'
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import numpy as np
import matplotlib.pyplot as plt

def generate_mesh(N):
    # 1. Create nodes
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    nodes = np.vstack([X.ravel(), Y.ravel()]).T
    
    # 2. Create elements (triangles)
    elements = []
    for j in range(N - 1): # loop over y
        for i in range(N - 1): # loop over x
            # Index of bottom-left corner of the cell
            idx = j * N + i
            
            # Triangle 1 (Lower Right)
            elements.append([idx, idx + 1, idx + N + 1])
            # Triangle 2 (Upper Left)
            elements.append([idx, idx + N + 1, idx + N])
            
    return nodes, np.array(elements)


# Example: 5x5 grid (16 squares, 32 triangles)
nodes, elements = generate_mesh(5)

# Visualization
plt.triplot(nodes[:,0], nodes[:,1], elements)
plt.plot(nodes[:,0], nodes[:,1], 'go', markersize=3) # Show nodes
plt.title(f"FEM Mesh: {len(elements)} Triangles")
plt.show()