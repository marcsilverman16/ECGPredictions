import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer


def lrap_loss(y_true_multi, gmm_probs):
    # Step 1: Prepare True Labels in a binary matrix format
    mlb = MultiLabelBinarizer()
    y_true_binary = mlb.fit_transform(y_true_multi)

    # Since GMM probabilities are already in the correct format (n_samples, n_clusters),
    # they can directly serve as the predicted "rankings" for LRAP calculation

    # Step 3: Calculate LRAP
    lrap_score = label_ranking_average_precision_score(y_true_binary, gmm_probs)
    return lrap_score


gmm_probs = np.array([[0.1, 0.7, 0.2], [0.4, 0.5, 0.1], [0.3, 0.3, 0.4]])
y_true_multi = [[1], [0, 1], [2]]



print(f"Label Ranking Average Precision (LRAP): {lrap_loss(y_true_multi, gmm_probs)}")
