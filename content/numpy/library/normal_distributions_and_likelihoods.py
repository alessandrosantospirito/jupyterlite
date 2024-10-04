import numpy as np
from scipy.stats import multivariate_normal

def calc_likelihoods_for_distributions_and_points(matrix_dist, matrix_points):
    """
    Calculate likelihoods for distributions and points.
    
    Arguments:
    matrix_dist: numpy.ndarray with shape (n, 3, 2). Each of the n-entries contains the mean and covariance for one distribution.
    matrix_points: numpy.ndarray with shape (m, 2). represents n points in 2-dimensional space.
    
    Returns:
    likelihoods: numpy.ndarray with shape (m, n) -- the likelihoods for each point to be observed by the a distribution
    """
    
    means = matrix_dist[:, 0]
    covariances = matrix_dist[:, 1:]
    likelihoods = np.array([multivariate_normal(mean=means[i], cov=covariances[i]).pdf(matrix_points) for i in range(len(means))])
    
    return likelihoods.T 

def calc_point_indices_to_distributions(distributions, points, treshhold_likelihood = 0):
    if points.shape[0] == 0:
        return np.zeros(distributions.shape[0]) - 1
    
    if points.shape[0] == 1 and distributions.shape[0] == 1:
        likelihood_entries = calc_likelihoods_for_distributions_and_points(distributions, points)
    
        if likelihood_entries < treshhold_likelihood: return np.array([-1])
        return np.array([0])

    likelihood_entries, likelihood_indicies_to_filter = calc_top_likelihood_of_points_to_distributions(distributions, points, treshhold_likelihood)

    return calc_point_indicies_to_likelihoods(likelihood_entries, likelihood_indicies_to_filter)

def calc_point_indicies_to_likelihoods_no_filter(likelihood_entries):
    cost_matrix = -1 * np.log(likelihood_entries)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    points_to_distributions = np.zeros(likelihood_entries.shape[1], dtype=row_ind.dtype) - 1
    points_to_distributions[col_ind] = row_ind

    return points_to_distributions

def calc_point_indicies_to_likelihoods(likelihood_entries, likelihood_indicies_to_filter):
    cost_matrix = -1 * np.log(likelihood_entries)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    points_to_distributions = np.zeros(likelihood_entries.shape[1], dtype=row_ind.dtype) - 1
    
    points_to_distributions[col_ind] = row_ind

    for point_index, distribution_index in enumerate(col_ind):
        if likelihood_indicies_to_filter[point_index, distribution_index]:
            points_to_distributions[point_index] = -1
    
    return points_to_distributions

def calc_top_likelihood_of_points_to_distributions(distributions, points, treshhold_likelihood=0):
    """
    Calculate the likelihood values of the top-s points relative to the distributions and the respective point-indicies
    
    Arguments:
    distributions: numpy.ndarray with shape (n, 3, 2). Each of the n-entries contains the mean and covariance for one distribution.
    points: numpy.ndarray with shape (m, 2). represents n points in 2-dimensional space.
    
    Returns:
    likelihood_entries: numpy.ndarray with shape (m, n) -- the likelihoods for each point to be observed by the a distribution
    likelihood_indicies_to_filter: numpy.ndarray with shape (m, n) of boolean value: point-likelihood below the treshhold
    """
    
    likelihoods = calc_likelihoods_for_distributions_and_points(distributions, points)
    sorted_indices = np.argsort(likelihoods, axis=0)
    ranks = np.zeros_like(likelihoods, dtype=int)
    n_rows, n_cols = likelihoods.shape
    ranks[sorted_indices, np.arange(n_cols)] = np.tile(np.arange(n_rows), (n_cols, 1)).T
    
    mask_binary = np.array((n_rows - ranks) <= n_cols, dtype=int)
    cumsum_array = np.cumsum(mask_binary, axis=0)
    
    s = min(n_rows, n_cols)
    extended_likelihood_entries = np.zeros((s+1, n_cols + 1))
    
    mask  = np.array((n_rows - ranks) <= n_cols)
    
    points_to_consider = np.where(np.any(mask, axis=1))[0]
    filtered_points = points[points_to_consider]
    
    likelihood_entries = calc_likelihoods_for_distributions_and_points(distributions, filtered_points)
    likelihood_indicies_to_filter = likelihood_entries < treshhold_likelihood
    # punish points that should be filtered out
    likelihood_entries[likelihood_indicies_to_filter] = 1e-50

    return likelihood_entries, likelihood_indicies_to_filter