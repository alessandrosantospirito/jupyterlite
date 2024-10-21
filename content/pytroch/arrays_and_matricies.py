import torch

def group_ordering_based_on_sorting(matrix, grp_idx=0, val_idx=1):
    """
    matrix: tensor([[ 1.,  2.], [ 0.,  3.], [ 1.,  0.]])
    b: tensor([[1., 0.], [0., 1.], [1., 2.]])
    c: tensor([[ 0.,  1.], [ 1.,  0.], [ 1.,  2.]])
    d: tensor([1, 0, 2])
    """

    b = torch.empty(matrix.shape[0], 2)
    b[:, 0] = matrix[:, 0]
    b[:, 1] = torch.arange(matrix.shape[0])
    c = sort_matrix_by_nth_entry(b)
    d = c[:, 1]

    return d

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

def reverse_sort_matrix_by_nth_entry(matrix, modified_matrix, n=0):
    sorted_values, sorted_indicies = torch.sort(matrix[:, n])
    _, reverse_sort_indicies = torch.sort(sorted_indicies)

    return modified_matrix[reverse_sort_indicies]

def reverse_sort_matrix_by_group(matrix, modified_matrix, grp_idx=0, val_idx=1):
    transposed = matrix.T
    sorted_indices = torch.argsort(transposed[grp_idx] * matrix.shape[grp_idx] + transposed[val_idx])
    reverse_sorted_indicies = torch.sort(sorted_indices)
    
    return modified_matrix[reverse_sorted_indicies]

def reverse_sort_matrix_by_nth_and_mth_column(matrix, modified_matrix, nth_col=0, mth_col=1):
    
    modified_matrix = reverse_sort_matrix_by_nth_entry(matrix, modified_matrix, nth_col)
    reverse_sorted_matrix = reverse_sort_matrix_by_group(matrix, modified_matrix, nth_col, mth_col)

    return reverse_sorted_matrix

def filter_columns_based_on_tensor(a, b, grp_idx = 0):
    """
    a = tensor([[0., 11.], [2., 3.], [1., 12.]])
    b = tensor([0., 2.])
    c = tensor([ True,  True, False]),
    b = tensor([[ 0., 11.], [ 2.,  3.]])
    """

    c = torch.isin(a[:, 0], b)
    d = a[c]

    return d

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
    a = tensor([[ 0.,  1.], [ 0.,  5.], [ 0.,  3.], [ 1.,  7.], [ 1.,  6.], [ 0.,  4.], [ 1., 10.], [ 1., 11.], [ 0.,  5.], [ 0.,  6.]])
    b = tensor([[ 0.,  6.], [ 1., 11.]])
    c = tensor([0, 0, 0, 1, 1, 0, 1, 1, 0, 0])
    d = tensor([[0., 0.], [1., 6.]])
    e = tensor([[ 0.,  1.], [ 0.,  5.], [ 0.,  3.], [ 1., 13.], [ 1., 12.], [ 0.,  4.], [ 1., 16.], [ 1., 17.], [ 0.,  5.], [ 0.,  6.]]
    """

    a = matrix
    b = mapping
    c = (a[:, 0][..., None] == b[:, 0]).nonzero(as_tuple=True)[1]
    d = torch.cumsum(b, dim=0)
    e = a
    e[:, 1] = e[:, 1] + d[c][:, 1]

    return e
