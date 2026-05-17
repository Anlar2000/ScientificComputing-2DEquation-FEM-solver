import numpy as np
from scipy.sparse.linalg import spsolve

def solve_heat_equation(K, M, U_initial, dt, num_steps):
    # The left-hand side matrix stays constant throughout time
    # (M + dt * K) * U_next = M * U_current
    LHS = M + dt * K
    
    # Store the results over time (optional, but good for plotting)
    U_history = [U_initial]
    U_current = U_initial.copy()
    
    for n in range(num_steps):
        # Calculate the right-hand side
        RHS = M.dot(U_current)
        
        # Solve the linear system for the next time step
        U_next = spsolve(LHS, RHS)
        
        U_history.append(U_next)
        U_current = U_next
        
    return U_history