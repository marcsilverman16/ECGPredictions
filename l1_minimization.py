#!/usr/bin/env python3

import numpy as np
import cvxpy as cp
import time

"Source: https://ieeexplore.ieee.org/abstract/document/8262793"
"Algorithm 6."

def find_broken_triangles(D):
    n = D.shape[0]
    broken_triangles = []
    for i in range(n):
        for j in range(i + 1, n):  
            for k in range(j + 1, n):  
                if D[i, j] > D[i, k] + D[k, j] or D[i, k] > D[i, j] + D[j, k] or D[j, k] > D[i, j] + D[i, k]:
                    broken_triangles.append((i, j, k))
                    
    return broken_triangles


def iteratively_reweighted_l1_minimization(D, iters=10, epsilon=1e-3):
    n = D.shape[0]
    W = np.ones((n, n))  
    D_hat = D.astype(np.float64).copy()

    for t in range(iters):
        P = cp.Variable((n, n), symmetric=True)
        objective = cp.Minimize(cp.sum(cp.multiply(W, cp.abs(P))))  # P is symmetric if D is symmetric
        constraints = [D_hat + P >= 0] # non-negative distances 

        broken_triangles = find_broken_triangles(D_hat)
        for i, j, k in broken_triangles:
            constraints += [
                D_hat[i, j] + P[i, j] <= D_hat[i, k] + P[i, k] + D_hat[k, j] + P[k, j],
                D_hat[i, k] + P[i, k] <= D_hat[i, j] + P[i, j] + D_hat[j, k] + P[j, k],
                D_hat[j, k] + P[j, k] <= D_hat[i, j] + P[i, j] + D_hat[i, k] + P[i, k]
            ]

        for i in range(n):
            constraints.append(P[i, i] == -D_hat[i, i])

        # ensuring diagonal of P makes D_hat's diagonal zero
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)  

        # updating D_hat and weights
        P_value = P.value
        D_hat += P_value
        W = 1 / (np.abs(P_value) + epsilon)

    np.fill_diagonal(D_hat, 0) # manually setting the diagonal of D_hat to zero to ensure metric property 

    return D_hat, W


def is_metric(D, tol=1e-5):
    """
    Checks if matrix D is a metric.
    """
    n = D.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if D[i, j] > D[i, k] + D[k, j] + tol:  # small numerical tolerance
                    return False, (i, j, k)
    return True, None


if __name__ == "__main__":
    n = 1000 

    # random dissimilarity matrix
    np.random.seed(42)  
    random_matrix = np.abs(np.random.randn(n, n))
    D = (random_matrix + random_matrix.T) / 2 # D is the given n x n dissimilarity matrix
    np.fill_diagonal(D, 0)

    print("Initial D is metric:", is_metric(D)[0]) # validating if D is metric before correction
    
    start_time = time.time()  
    D_hat, W = iteratively_reweighted_l1_minimization(D)
    end_time = time.time()  

    print("Corrected Distance Matrix D_hat:\n", D_hat)
    print("Corrected D_hat is metric:", is_metric(D_hat)[0])
    print(f"Execution time: {end_time - start_time:.2f} seconds") # validating D_hat is metric after correction


## Previous Implementation
# def iteratively_reweighted_l1_minimization(D, iters=10, epsilon=1e-3):
#     n = D.shape[0]
#     W = np.ones((n, n))  
#     D_hat = D.astype(np.float64).copy()

#     for t in range(iters):
#         P = cp.Variable((n, n), symmetric=True)  # P is symmetric if D is symmetric
#         objective = cp.Minimize(cp.sum(cp.multiply(W, cp.abs(P))))
#         constraints = [D_hat + P >= 0]  # non-negative distances 

#         # ensuring triangle inequality
#         for i in range(n):
#             for j in range(n):
#                 for k in range(n):
#                     if i != j and i != k and j != k:
#                         constraints.append(D_hat[i, j] + P[i, j] <= D_hat[i, k] + P[i, k] + D_hat[k, j] + P[k, j])

#         # ensuring diagonal of P makes D_hat's diagonal zero
#         for i in range(n):
#             constraints.append(P[i, i] == -D_hat[i, i])

#         prob = cp.Problem(objective, constraints)
#         prob.solve()

#         # updating D_hat and weights
#         P_value = P.value
#         D_hat += P_value
#         W = 1 / (np.abs(P_value) + epsilon)

#     np.fill_diagonal(D_hat, 0) # manually setting the diagonal of D_hat to zero to ensure metric property

#     return D_hat, W


# def is_metric(D):
#     """
#     checks if matrix D is a metric.
#     """
#     n = D.shape[0]
#     for i in range(n):
#         for j in range(n):
#             for k in range(n):
#                 if D[i, j] > D[i, k] + D[k, j]:
#                     return False, (i, j, k)
#     return True, None



# if __name__ == "__main__":
#     n = 3  
#     D = np.array([[6, 2, 1], [2, 1, 4], [1, 4, 2]], dtype=np.float64)  # D is the given n x n dissimilarity matrix
#     print("Initial D is metric:", is_metric(D)[0])  # validating if D is metric before correction
    
#     D_hat, W = iteratively_reweighted_l1_minimization(D)
    
#     print("Corrected Distance Matrix D_hat:\n", D_hat)
#     # print("Final Weight Matrix W:\n", W)
#     print("Corrected D_hat is metric:", is_metric(D_hat)[0])  # validating if D_hat is metric after correction l