import cvxpy as cp
import numpy as np

SOLVER_EPSILON = 1e-4


def single_shot_COAP(
    X, endowments, resource_values, i, community_indices, neighbours
) -> np.ndarray:
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
    x_i = cp.Variable(n, nonneg=True)
    static = X.T @ resource_values - X[i] * resource_values[i]

    # Epsilon is added to prevent log of (near) zero
    static_received = cp.sum(
        [
            (endowments[j] * resource_values[j])
            * cp.log(static[j] + x_i[j] * resource_values[i] + SOLVER_EPSILON)
            for j in community_indices
        ]
    )

    constraints = [
        cp.sum(x_i) <= endowments[i],  # x_i must be within endowments
        x_i >= 0,  # nonnegative allocation (might be redundant since x_1: nonneg=True)
    ] + [
        x_i[j] <= SOLVER_EPSILON for j in range(n) if j not in neighbours
    ]  # Allocations to agents not in the neighbourhood of i must be zero
    problem = cp.Problem(cp.Maximize(static_received), constraints)

    try:
        problem.solve(
            solver=cp.CLARABEL,
            max_iter=100,  # Maximum iterations
            time_limit=1.0,  # Time limit in seconds
            verbose=False,
        )
    except Exception:
        problem.solve(solver=cp.SCS, eps=SOLVER_EPSILON, max_iters=100, verbose=False)

    return x_i.value if problem.status in ["optimal", "optimal_inaccurate"] else None


def make_fixed_agent_coap_solver(
    n, i, community_indices, neighbours, endowments, resource_values
):
    """Optimized factory creating dedicated solver for single-agent repeated COAP optimization.

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
    """
    # Map market ids to local ids for neighbours
    neighbour_array = np.array(sorted(neighbours))
    num_neigh = len(neighbour_array)
    neighbour_to_local_idx = {idx: j for j, idx in enumerate(neighbour_array)}

    community_array = np.array(list(community_indices))

    # Find community members which are also neighbours
    community_members_in_neighbourhood_mask = np.isin(community_array, neighbour_array)
    community_members_in_neighbourhood = community_array[
        community_members_in_neighbourhood_mask
    ]

    # Map market ids to local ids
    community_local_idx = np.array(
        [neighbour_to_local_idx[c] for c in community_members_in_neighbourhood]
    )

    # Allocations to nieghbours. Not to all agents.
    x_i_neighbours = cp.Variable(num_neigh, nonneg=True)

    # Parameter for dynamic static values
    static_val_param = cp.Parameter(len(community_members_in_neighbourhood))

    # Pre-compute constants
    community_weights = cp.Constant(
        endowments[community_members_in_neighbourhood]
        * resource_values[community_members_in_neighbourhood]
    )
    community_resource_values = cp.Constant(
        resource_values[community_members_in_neighbourhood]
    )

    objective = cp.sum(
        cp.multiply(
            community_weights,
            cp.log(
                static_val_param
                + cp.multiply(
                    x_i_neighbours[community_local_idx], community_resource_values
                )
                + SOLVER_EPSILON
            ),
        )
    )

    # Constraint: total allocation to neighbors cannot exceed agent's endowment
    constraints = [cp.sum(x_i_neighbours) <= endowments[i]]
    prob = cp.Problem(cp.Maximize(objective), constraints)

    def solve(X):
        # Compute static received utility for all agents based on X
        received_utility = X.T @ resource_values - X[i] * resource_values
        # Extract static values for neighbors
        received_utility_neighbours = received_utility[
            community_members_in_neighbourhood
        ]

        # Update parameter
        static_val_param.value = received_utility_neighbours

        try:
            prob.solve(cp.CLARABEL, max_iter=200, time_limit=1.0, verbose=False)
        except Exception:
            prob.solve(cp.SCS, eps=SOLVER_EPSILON, max_iters=200, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return None

        full_solution = np.zeros(n)
        full_solution[neighbour_array] = x_i_neighbours.value

        return full_solution

    return solve
