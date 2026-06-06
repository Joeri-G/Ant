import cvxpy as cp
import numpy as np

SOLVER_EPSILON = 1e-6

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
           max_iter=2000,         # Maximum iterations
           verbose=False,         # Suppress output
           time_limit=5.0)        # Time limit in seconds
    except Exception:
        problem.solve(solver=cp.SCS, eps=1e-7, max_iters=5000, verbose=False)

    return (x_i.value if problem.status in ['optimal', 'optimal_inaccurate'] else None)


class ParameterizedCOAP:
    """
    Parameterized optimizer for COAP convex optimization problems.
    
    Problem structure remains constant; only X, i, neighbours, community_indices change.
    resource_values and endowments are constant across solves.
    """

    def __init__(self, n, endowments, resource_values, eps=1e-9):
        """
        Initialize with constant structural parameters.
        
        Parameters:
        -----------
        n : int
            Number of agents
        endowments : np.ndarray
            Vector of length n. Constant max distributable resources per agent.
        resource_values : np.ndarray
            Vector of length n. Constant value of each agent's resource type.
        eps : float
            Small epsilon for numerical stability in log
        """
        self.n = n
        self.endowments = np.array(endowments)
        self.resource_values = np.array(resource_values)
        self.eps = eps
        
        # Variable (same across solves)
        self.x_i = cp.Variable(n, nonneg=True)
        
        # Parameters that change between solves
        self.X_param = cp.Parameter((n, n))
        self.center_idx_param = cp.Parameter(integer=True)
        self.community_param = cp.Parameter(n, boolean=True)  # mask for community members
        self.neighbour_param = cp.Parameter(n, boolean=True)  # mask for reachable neighbors
        
        self.problem = self._build_problem()
    
    def _build_problem(self):
        """Build CVXPY problem structure once."""
        # Static received from others excluding center agent's contribution
        static = cp.matmul(self.X_param.T, self.resource_values) - \
                 self.X_param[self.center_idx_param] * self.resource_values[self.center_idx_param]
        
        # Objective: sum over community members only (using mask)
        objective = cp.sum([
            (self.endowments[j] * self.resource_values[j]) * 
            cp.log(static[j] + self.x_i[j] * self.resource_values[self.center_idx_param] + self.eps)
            for j in range(self.n) if self.endowments[j] > 0 or True  # Will use mask below
        ])
        
        # Actually use mask properly:
        # Rebuild with proper masking using element-wise operations
        community_mask = self.community_param
        neighbour_mask = self.neighbour_param
        
        terms = []
        for j in range(self.n):
            # Use condition that term is included if j is in community
            pass  # We'll do explicit summation for clarity
        
        # Simpler approach: build dynamically in solve()
        return None  # Rebuild needed due to dynamic community/neighbour
    
    def solve(self, X, i, community_indices, neighbours):
        """
        Solve optimization with new variable parameters.
        
        Parameters:
        -----------
        X : np.ndarray
            Current market allocation matrix (n x n). X[a,b] = amount FROM a TO b.
        i : int
            Index of center agent whose allocation we optimize.
        community_indices : list or np.array
            List of agents in community C^k_i (must include 'i' and neighbors).
        neighbours : list or np.array
            List of agents in neighbourhood N_i.
            
        Returns:
        --------
        x_new : np.ndarray
            Optimal allocation vector for agent i (length n).
            Returns None if optimization fails.
        """
        # Build fresh problem due to changing community/neighbour structure
        static = self.X_param @ self.resource_values - \
                 self.X_param[i] * self.resource_values[i]
        
        objective = cp.sum([
            (self.endowments[j] * self.resource_values[j]) * 
            cp.log(static[j] + self.x_i[j] * self.resource_values[i] + self.eps)
            for j in community_indices
        ])
        
        non_neighbours = [j for j in range(self.n) if j not in neighbours]
        constraints = [
            cp.sum(self.x_i) <= self.endowments[i],
            self.x_i >= 0,
        ] + [self.x_i[j] <= self.eps for j in non_neighbours]
        
        self.X_param.value = X
        
        prob = cp.Problem(cp.Maximize(objective), constraints)
        prob.solve(verbose=False)
        
        return (self.x_i.value if prob.status in ['optimal', 'optimal_inaccurate'] else None)