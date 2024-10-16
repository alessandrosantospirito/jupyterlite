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
    num_groups = int(matrix[:, grp_idx].max().item()) + 1

    result = torch.zeros(num_groups, 2)
    
    result[:, grp_idx] = torch.arange(num_groups)
    
    result[:, val_idx] = torch.scatter_reduce(
        input=torch.zeros(num_groups),
        dim=0,
        index=matrix[:, grp_idx].long(),
        src=matrix[:, val_idx],
        reduce='amax'
    )

    return result
