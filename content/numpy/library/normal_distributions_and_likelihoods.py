import numpy as np
from scipy.stats import multivariate_normal

def calc_likelihoods_for_distributions_and_points(matrix_dist, matrix_points):
    assert isinstance(matrix_dist, np.ndarray), "matrix_dist must be a numpy array"
    assert isinstance(matrix_points, np.ndarray), "matrix_points must be a numpy array"
    assert matrix_dist.ndim == 3 and matrix_dist.shape[1:] == (3, 2), "matrix_dist must be of shape (m, 3, 2)"
    assert matrix_points.ndim == 2 and matrix_points.shape[1] == 2, "matrix_points must be of shape (n, 2)"

    means = matrix_dist[:, 0]
    covariances = matrix_dist[:, 1:]
    likelihoods = np.array([multivariate_normal(mean=means[i], cov=covariances[i]).pdf(matrix_points) for i in range(len(means))])
    
    assert isinstance(likelihoods, np.ndarray), "likelihoods must be a numpy array"
    assert likelihoods.shape == (matrix_dist.shape[0], matrix_points.shape[0]), "likelihoods must be of shape (m, n)"

    return likelihoods.T 

def calc_point_indices_to_distributions(distributions, points, threshold_likelihood = 0):
    if points.shape[0] == 0:
        return np.zeros(distributions.shape[0]) - 1
    
    if points.shape[0] == 1 and distributions.shape[0] == 1:
        likelihood_entries = calc_likelihoods_for_distributions_and_points(distributions, points)
    
        if likelihood_entries < threshold_likelihood: return np.array([-1])
        return np.array([0])

    likelihood_entries, likelihood_indices_to_filter = calc_top_likelihood_of_points_to_distributions(distributions, points, threshold_likelihood)

    return calc_point_indices_to_likelihoods(likelihood_entries, likelihood_indices_to_filter)

def calc_point_indices_to_likelihoods_no_filter(likelihood_entries):
    cost_matrix = -1 * np.log(likelihood_entries)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    points_to_distributions = np.zeros(likelihood_entries.shape[1], dtype=row_ind.dtype) - 1
    points_to_distributions[col_ind] = row_ind

    return points_to_distributions

def calc_point_indices_to_likelihoods(likelihood_entries, likelihood_indices_to_filter):
    cost_matrix = -1 * np.log(likelihood_entries)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    points_to_distributions = np.zeros(likelihood_entries.shape[1], dtype=row_ind.dtype) - 1
    
    points_to_distributions[col_ind] = row_ind

    for point_index, distribution_index in enumerate(col_ind):
        if likelihood_indices_to_filter[point_index, distribution_index]:
            points_to_distributions[point_index] = -1
    
    return points_to_distributions

def calc_top_likelihood_of_points_to_distributions(distributions, points, threshold_likelihood=0):
    assert isinstance(distributions, np.ndarray), "distributions must be a numpy array"
    assert isinstance(points, np.ndarray), "points must be a numpy array"
    assert distributions.ndim == 3 and distributions.shape[1:] == (3, 2), "distributions must be of shape (m, 3, 2)"
    assert points.ndim == 2 and points.shape[1] == 2, "points must be of shape (n, 2)"

    likelihoods = calc_likelihoods_for_distributions_and_points(distributions, points)
    ranks = np.argsort(np.argsort(-likelihoods, axis=0), axis=0)

    s = min(*likelihoods.shape)
    top_s_indices = ranks < s

    selected_points_mask = np.any(top_s_indices, axis=1)
    filtered_points = points[selected_points_mask]

    likelihood_entries = calc_likelihoods_for_distributions_and_points(distributions, filtered_points)
    likelihood_indices_to_filter = likelihood_entries < threshold_likelihood

    likelihood_entries[likelihood_indices_to_filter] = 1e-50

    assert isinstance(likelihood_entries, np.ndarray), "likelihood_entries must be a numpy array"
    assert isinstance(likelihood_indices_to_filter, np.ndarray), "likelihood_indices_to_filter must be a numpy array"
    assert likelihood_entries.shape[1] == distributions.shape[0], "likelihood_entries must have shape (s, m)"
    assert likelihood_indices_to_filter.shape == likelihood_entries.shape, "likelihood_indices_to_filter must be the same shape as likelihood_entries"

    return likelihood_entries, likelihood_indices_to_filter

path = "/workspaces/jupyterlite/content/numpy/html/02-calc_likelihoods_for_distributions_and_points.html"
with open(path, 'r', encoding='utf-8') as file:
    calc_likelihoods_for_distributions_and_points.__html__ = file.read()

path = "/workspaces/jupyterlite/content/numpy/html/02-calc_top_likelihood_of_points_to_distributions.html"
with open(path, 'r', encoding='utf-8') as file:
    calc_top_likelihood_of_points_to_distributions.__html__ = file.read()
