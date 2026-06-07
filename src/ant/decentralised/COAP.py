import cvxpy as cp
import numpy as np

SOLVER_EPSILON = 1e-4

def single_shot_COAP(X, endowments, resource_values, i, community_indices, neighbours) -> np.ndarray:
    """
    Calculates the optimal allocation vector x_i for agent i that maximizes 
    the Community Utility U^k_i, given a fixed set of community members.
    
    This function assumes a sequential market model where all other agents' 
    allocations remain static (x_j(t+1) = x_j(t) for j != i).

    Parameters:
    -----------
    X : np.ndarray
        Current market allocation matrix (n x n). 
        X[a, b] = amount allocated FROM agent a TO agent b.
    endowments : np.ndarray
        Vector of length n. endowments[i] is the max distributable resources for agent i.
    resource_values : np.ndarray
        Vector of length n. resource_values[j] is the value of agent j's resource type.
    i : int
        The index of the center agent whose allocation we are optimizing.
    community_indices : list or np.array
        List of integers representing the agents in the community C^k_i.
        Must include 'i' and any neighbors of 'i'.
    neighbours : list or np.array
        List of integers representing the agents in the neighbourhood N_i.

    Returns:
    --------
    x_new : np.ndarray
        The new optimal allocation vector for agent i (length n).
        Non-zero values will only appear at indices present in community_indices 
        AND reachable from i (logically enforced by the problem structure).
    """
    n = len(endowments)
    # Initializes problem size and creates a non-negative optimization variable vector for the agent's allocations.
    x_i = cp.Variable(n, nonneg=True)
    # Calculates the weighted resources already received by all agents from everyone except the center agent 'i'.
    static = X.T @ resource_values - X[i] * resource_values[i]

    # Constructs the objective function: sum of (endowment * value * log(total received)) for all agents in the community.
    # Epsilon is added to prevent log of (near) zero
    static_received = cp.sum([(endowments[j] * resource_values[j]) * cp.log(static[j] + x_i[j] * resource_values[i] + SOLVER_EPSILON) for j in community_indices])

    constraints = [
        cp.sum(x_i) <= endowments[i], # x_i must be within endowments
        x_i >= 0, # nonnegative allocation (might be redundant since x_1: nonneg=True)
        
    ] + [x_i[j] <= SOLVER_EPSILON for j in range(n) if j not in neighbours] # Allocations to agents not in the neighbourhood of i must be zero
    # Formulates the convex optimization problem to maximize the objective under endowment and non-negativity constraints.
    problem = cp.Problem(cp.Maximize(static_received), constraints)

    # Solves the problem using the SCS solver without printing output logs.
    # problem.solve(solver=cp.SCS, verbose=False)
    # problem.solve(solver=cp.SCS, eps=1e-7, max_iters=5000, verbose=False, normalize=True)
    try:
        problem.solve(solver=cp.CLARABEL,
           tol_gap_abs=1e-8,      # Absolute duality gap tolerance
           tol_gap_rel=1e-8,      # Relative duality gap tolerance  
           tol_feas=1e-8,         # Feasibility tolerance
           max_iter=100,          # Maximum iterations
           verbose=False,         # Suppress output
           time_limit=1.0)        # Time limit in seconds
    except Exception:
        problem.solve(solver=cp.SCS, eps=SOLVER_EPSILON, max_iters=100, verbose=False)

    return (x_i.value if problem.status in ['optimal', 'optimal_inaccurate'] else None)

def make_fixed_agent_coap_solver(n, i, community_indices, neighbours, endowments, resource_values):
    """Factory creating dedicated solver for single-agent repeated COAP optimization.
    
    All problem structure (agent index, community, neighborhood) is fixed at compile time.
    Only X and its derived 'static' terms update between solves—maximizing efficiency.
    
    Parameters:
    -----------
    n : int
        Total number of agents in market.
    i : int
        Static center agent being optimized.
    community_indices : array_like
        Agents in community C^k_i (includes i).
    neighbours : array_like
        Agents reachable from i (non-zero allocations allowed).
    endowments : np.ndarray
        Distributable resources per agent.
    resource_values : np.ndarray
        Value per unit of each agent's resource type.
    
    Returns:
    --------
    solve : Callable[[np.ndarray], np.ndarray|None]
        Takes current allocation matrix X, returns optimal x_i for agent i.
    """
    x_i = cp.Variable(n, nonneg=True)
    static_val = cp.Parameter(n)
    
    # Pre-compute constant weights for community objective
    community_array = np.array(list(community_indices))
    community_resource_values = cp.Constant(resource_values[community_array])
    community_weights = cp.Constant(endowments[community_array] * resource_values[community_array])
    
    # Constraint masks precomputed (one-time cost)
    neighbour_mask = cp.Constant(np.isin(np.arange(n), list(neighbours)).astype(float))
    endowment_limit = cp.Constant(endowments[i])
    
    obj = cp.sum(cp.multiply(community_weights, cp.log(static_val[community_array] + cp.multiply(x_i[community_array], community_resource_values) + SOLVER_EPSILON)))
    cons = [
        cp.sum(x_i) <= endowment_limit,
        cp.multiply(x_i, 1 - neighbour_mask) <= 0  # Force zero allocation to non-neighbors (elementwise)
    ]
    prob = cp.Problem(cp.Maximize(obj), cons)
    
    def solve(X):
        static_val.value = X.T @ resource_values - X[i] * resource_values[i]
        
        try:
            prob.solve(cp.CLARABEL, tol_gap_abs=1e-8, tol_gap_rel=1e-8,
                       tol_feas=1e-8, max_iter=100, verbose=False, time_limit=1.0)
        except: prob.solve(cp.SCS, eps=SOLVER_EPSILON, max_iters=200, verbose=False)
        
        return x_i.value if prob.status in ['optimal', 'optimal_inaccurate'] else None
    
    return solve