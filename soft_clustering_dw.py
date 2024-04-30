import numpy as np
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
from l1_minimization import iteratively_reweighted_l1_minimization, is_metric


def soft_assignment_gmm(dm, n_clusters, n_dim = 2):
    mds = MDS(n_components=n_dim, dissimilarity='precomputed', random_state=0)
    X_transformed = mds.fit_transform(dm)

    # Now apply GMM
    gmm = GaussianMixture(n_clusters, random_state=0)
    gmm.fit(X_transformed)
    soft_assignments = gmm.predict_proba(X_transformed)
    return soft_assignments, X_transformed


def soft_assignment_fuzzy(dm, n_clusters, n_iter = 1000, epsilon=1e-6):
    n_points = dm.shape[0]
    
    # Initialize random cluster centers
    cluster_centers_indices = np.random.choice(range(n_points), size=n_clusters, replace=False)
    
    # Initialize membership scores matrix with zeros
    membership_scores = np.zeros((n_points, n_clusters))
    
    # Initial rough assignment based on minimal dissimilarity
    for i in range(n_points):
        for j in range(n_clusters):
            membership_scores[i, j] = dm[i, cluster_centers_indices[j]]
    
    # Add a small constant epsilon to avoid division by zero
    membership_scores += epsilon
    
    # Normalize scores to get a sort of probability distribution for each point across clusters
    membership_scores = 1 / membership_scores  # Invert scores because lower dissimilarity means closer association
    membership_scores = membership_scores / membership_scores.sum(axis=1, keepdims=True)
    
    # Iteratively update cluster centers and membership scores
    for iteration in range(n_iter): 
        for i in range(n_points):
            for j in range(n_clusters):
                membership_scores[i, j] = dm[i, cluster_centers_indices[j]] + epsilon
        membership_scores = 1 / membership_scores
        membership_scores = membership_scores / membership_scores.sum(axis=1, keepdims=True)
    
    return membership_scores

if __name__ == "__main__":
    n = 100
    np.random.seed(42)  
    random_matrix = np.abs(np.random.randn(n, n))
    D = (random_matrix + random_matrix.T) / 2 # D is the given n x n dissimilarity matrix
    np.fill_diagonal(D, 0)
    
    D_hat, W = iteratively_reweighted_l1_minimization(D)
    n_clusters = 5

    print("Soft clustering GMM:",soft_assignment_gmm(D_hat, n_clusters))
    print("\n")
    print("Soft clustering Fuzzy:",soft_assignment_fuzzy(D_hat, n_clusters))