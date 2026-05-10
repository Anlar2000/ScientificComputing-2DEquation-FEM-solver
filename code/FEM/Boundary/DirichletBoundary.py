import numpy as np

def apply_boundary_conditions(K, M, nodes):
    # 1. Identify boundary nodes (e.g., all edges of the [0,1] square)
    x, y = nodes[:, 0], nodes[:, 1]
    boundary_indices = np.where((x == 0) | (x == 1) | (y == 0) | (y == 1))[0]
    
    K = K.tolil()
    M = M.tolil()
    
    for idx in boundary_indices:
        # Zero out the row in K and put a 1 on the diagonal
        K[idx, :] = 0
        K[idx, idx] = 1
        
        # Zero out the row in M (boundary nodes don't evolve via the mass matrix)
        M[idx, :] = 0
        M[idx, idx] = 0 # Or 1 depending on your time-stepping scheme
        
    return K.tocsr(), M.tocsr()