import torch

def sort_matrix_by_nth_entry(matrix, n=0):
    sorted_values, sorted_indices = torch.sort(matrix[:, n])

    return  matrix[sorted_indices]

def sort_matrix_by_group(matrix, grp_idx=0, val_idx=1):
    transposed = matrix.T
    sorted_indices = torch.argsort(transposed[grp_idx] * matrix.shape[grp_idx] + transposed[val_idx])
    
    return matrix[sorted_indices]

def sort_matrix_by_nth_and_mth_column(matrix, nth_col=0, mth_col=1):
    """
    First priority is nth-column, second priority is mth-column
    """

    matrix = sort_matrix_by_group(matrix, nth_col, mth_col)
    matrix = sort_matrix_by_nth_entry(matrix, nth_col)

    return matrix

def maximum_value_by_grp(matrix, grp_idx=0, val_idx=1):
    """
    matrix: torch.tensor([[ 0.,  2.], [ 0.,  3.], [ 1.,  0.], [ 1.,  3.], [ 1.,  4.], [ 2.,  23.], [ 3.,  4.], [ 0.,  7.], [ 1.,  0.], [ 1.,  3.], [ 1.,  7.], [ 0., 11.]])
    result: tensor([[ 0., 11.], [ 1.,  7.], [ 2., 23.], [ 3.,  4.]])
    """
    
    unique_grps = torch.unique(matrix[:, grp_idx])
    count_unique_grps = unique_grps.shape[0]
    result = torch.zeros(count_unique_grps, 2)
    result[:, grp_idx] = unique_grps[:]
    indices = (matrix[:, grp_idx][..., None] == unique_grps).nonzero(as_tuple=True)[1]
    
    result[:, val_idx] = torch.scatter_reduce(
        input=torch.zeros(count_unique_grps),
        dim=0,
        index=indices.long(),
        src=matrix[:, val_idx],
        reduce='amax'
    )
    
    return result

def add_to_matrix_mapping_values(matrix, mapping, grp_idx = 0, val_idx = 1):
    """
    a = tensor([[ 0.,  1.], [ 0.,  5.], [ 0.,  3.], [ 0.,  5.], [ 0.,  6.], [ 1.,  7.], [ 1.,  6.], [ 1., 11.], [ 5.,  4.], [ 7., 10.]])
    b = tensor([[ 0.,  0.], [ 1.,  7.], [ 5., 12.], [ 7.,  5.]])
    c = tensor([0, 1, 0, 1, 1, 0, 1, 1, 0, 0])
    d = tensor([[ 0.,  0.], [ 1.,  7.], [ 6., 19.], [13., 24.]])
    e = tensor([[ 0.,  1.], [ 1., 12.], [ 0.,  3.], [ 1., 14.], [ 1., 13.], [ 0.,  4.], [ 1., 17.], [ 1., 18.], [ 0.,  5.], [ 0.,  6.]]))
    """

    a = matrix
    b = mapping
    c = (a[:, 0][..., None] == b[:, 0]).nonzero(as_tuple=True)[1]
    d = torch.cumsum(b, dim=0)
    e = a
    e[:, 1] = e[:, 1] + d[c][:, 1]
    
    return e
