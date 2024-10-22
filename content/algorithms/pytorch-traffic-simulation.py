import torch

def fill_zero_places(a):
    """
    a = tensor([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    b = tensor([-1,  1,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0, 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0])
    nonzero_indices = tensor([ 0,  1,  6,  8, 14, 23])
    diff = tensor([1, 5, 2, 6, 9, 8])
    c = tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    d = tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    """
    b = torch.cat([torch.tensor([-1], device=a.device), a])
    nonzero_indices = torch.nonzero(b).squeeze()
    diff = torch.diff(torch.cat([nonzero_indices, torch.tensor([b.shape[0]], device=b.device)]))
    c = b[nonzero_indices].repeat_interleave(diff)
    c[c < 0] = 0
    d = c[1:]
    
    return d

def shift_values_to_next_available_zero(a):
    """
    a = tensor([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    b = tensor([1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    c = tensor([ 0,  1,  0,  0,  0,  0,  6,  0,  8,  0,  0,  0,  0,  0, 14,  0,  0,  0, 0,  0,  0,  0,  0, 23,  0,  0,  0,  0,  0,  0,  0])
    d = tensor([ 1,  0,  0,  0,  0,  6,  0,  8,  0,  0,  0,  0,  0, 14,  0,  0,  0,  0, 0,  0,  0,  0, 23,  0,  0,  0,  0,  0,  0,  0])
    e = tensor([ 0,  1,  1,  1,  1,  1,  6,  6,  8,  8,  8,  8,  8,  8, 14, 14, 14, 14, 14, 14, 14, 14, 14, 23, 23, 23, 23, 23, 23, 23])
    f = tensor([1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    """
    b = (a > 0).int()
    b = torch.cat([torch.tensor([1], device=a.device), b])
    c = torch.arange(b.shape[0], device=a.device) * b
    d = c[1:]
    e = torch.roll(fill_zero_places(d), 1)
    e[0] = 0
    f = torch.zeros_like(a)
    f.scatter_add_(0, e.long(), a)
        
    return f

def fill_zero_values_ascending_rest_with_zero(a):
    """
    a = tensor([4, 3, 2, 1, 0, 1, 0, 5, 4, 3, 2, 1, 0, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 0, 4, 3, 2, 1, 0, 0])
    b = tensor([1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0], dtype=torch.int32)
    c = tensor([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 7])
    d = tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 2])
    """
    b = (a > 0).int()
    c = torch.cumsum(1 - b, dim=0)
    d = (c * (1 - b) - torch.maximum(b * c, torch.zeros_like(c)).cummax(dim=0)[0]) * (1 - b)
    
    return d

def find_one_zero_pattern(a):
    """
    a = tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    b = tensor([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=torch.int32)
    c = tensor([ 0,  0,  0, -1,  1, -1,  1,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0, 0,  0, -1,  1, -1,  1,  0,  0,  0,  0,  0, -1], dtype=torch.int32)
    d = tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    """
    b = torch.roll(a.int(), -1)
    b[-1] = 1
    c = a - b
    d = (c == 1).int()
    
    return d

def fill_zero_values_descending_rest_with_zero(a):
    b = fill_zero_values_ascending_rest_with_zero(a)
    c = (b > 0).int()
    d = find_one_zero_pattern(c)
    e = d * b
    f = shift_values_to_next_available_zero(e)
    g = fill_zero_places(f)
    h = ((g + 1) - b) * c
    
    return h

def fill_zero_values_descending_rest_with_zero(a):
    """
    a = tensor([4, 3, 2, 1, 0, 1, 0, 5, 4, 3, 2, 1, 0, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 0, 4, 3, 2, 1, 0, 0])
    b = tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 2])
    c = tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1], dtype=torch.int32)
    d = tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    e = tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    f = tensor([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    g = tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    h = tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0])
    """
    b = fill_zero_values_ascending_rest_with_zero(a)
    c = (b > 0).int()
    d = find_one_zero_pattern(c)
    e = d * b
    f = shift_values_to_next_available_zero(e)
    g = fill_zero_places(f)
    h = ((g + 1) - b) * c
    
    return h

def find_values_that_must_be_reordered(a):
    """
    a = tensor([ 3, 10,  8, 11,  5, 13,  7, 14, 18, 15, 19, 20, 13, 21, 16, 24, 19, 24, 23, 22, 28, 22, 27, 26, 33, 32, 30, 32, 31, 36])
    b = tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    c = tensor([2, 8, 5, 7, 0, 7, 0, 6, 9, 5, 8, 8, 0, 7, 1, 8, 2, 6, 4, 2, 7, 0, 4, 2, 8, 6, 3, 4, 2, 6])
    d = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 6])
    e = tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1], dtype=torch.int32)
    f = tensor([1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0], dtype=torch.int32)
    """
    b = torch.arange(1, a.shape[0] + 1)
    c = a - b
    d, _ = torch.cummin(c.flip(0), dim=0)
    d = d.flip(0)
    e = (c == d).int()
    f = 1 - e

    return f

def adjust_cars_no_lane_change(a):
    """
    a = tensor([ 3, 10,  8, 11,  5, 13,  7, 14, 18, 15, 19, 20, 13, 21, 16, 24, 19, 24, 23, 22, 28, 22, 27, 26, 33, 32, 30, 32, 31, 36, -1, -1])
    b = tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1], dtype=torch.int32)
    c = tensor([4, 3, 2, 1, 0, 1, 0, 5, 4, 3, 2, 1, 0, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 0, 4, 3, 2, 1, 0, 0, 1, 0])
    d = tensor([ 0,  0,  0,  0,  5,  0,  7,  0,  0,  0,  0,  0, 13,  0,  0,  0,  0,  0, 0,  0,  0, 22,  0, 26,  0,  0,  0,  0, 31, 36, 0, -1])
    e = tensor([ 5,  0,  0,  0,  0,  7,  0, 13,  0,  0,  0,  0,  0, 22,  0,  0,  0,  0, 0,  0,  0,  0, 26,  0, 31,  0,  0,  0,  0, 36, -1, 0])
    f = tensor([ 5,  5,  5,  5,  5,  7,  7, 13, 13, 13, 13, 13, 13, 22, 22, 22, 22, 22, 22, 22, 22, 22, 26, 26, 31, 31, 31, 31, 31, 36, 0, 0])
    g = tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 36])
    """

    b = 1 - find_values_that_must_be_reordered(a)
    # Extend a and b with dummy-values to cover the cases where no car has to be reordered without throwing an error
    a = torch.cat([a, torch.tensor([-1, -1])])
    b = torch.cat([b, torch.tensor([0, 1])])
    c = fill_zero_values_descending_rest_with_zero(b)
    d = a * b
    e = shift_values_to_next_available_zero(d)
    f = fill_zero_places(e)
    g = f[:-2] - c[:-2]

    return g

def mapping_max_values_to_prev_lanes(cars):
    """
    - two lanes
    """
    max_values_per_lane = maximum_value_by_grp(cars, 0, 1)
    mapping_max_values_per_lane = torch.clone(max_values_per_lane)
    mapping_max_values_per_lane[1:, 1] = mapping_max_values_per_lane[:-1, 1]
    mapping_max_values_per_lane[0, 1] = 0

    return mapping_max_values_per_lane

def indicies_where_the_value_changes(a):
    """
    question: has value of position i a different value then position i-1?
    - the edge case for the first position is decidided as True
    
    a = tensor([0, 0, 0, 1, 0, 2, 0, 3])
    b = tensor([ 0,  0,  1, -1,  2, -2,  3])
    c = tensor([ True, False, False,  True,  True,  True,  True,  True])
    d = tensor([0, 3, 4, 5, 6, 7])
    """
    b = a[1:,] - a[:-1]
    c = (b != 0)
    c = torch.cat([torch.tensor([True]), c])
    d = (c==True).nonzero(as_tuple=True)[0]

    return d

def put_slowest_car_infront(cars, track_sizes, grp_idx = 0, val_idx = 1):
    """
    asserts that cars are sorted by grp_idx
    """
    # put the lowest cars of each lane infront of the top car
    indicies_slowest_cars = indicies_where_the_value_changes(cars[:, grp_idx])
    used_lanes = torch.unique(cars[:, grp_idx])
    slowest_cars = cars[indicies_slowest_cars]
    track_sizes_used_lanes = filter_columns_based_on_tensor(track_sizes, used_lanes)
    slowest_cars[:, val_idx] = slowest_cars[:, val_idx] + track_sizes_used_lanes[:, val_idx]
    cars_with_slowest_car_infront = torch.cat([cars, slowest_cars])
    indicies_of_artifical_cars = torch.cat([indicies_slowest_cars[1:], torch.tensor([cars.shape[0]])]) + torch.arange(0, indicies_slowest_cars.shape[0])
    mask_valid_cars = torch.ones(cars_with_slowest_car_infront.shape[0], dtype=torch.bool)
    mask_valid_cars[indicies_of_artifical_cars] = False
    cars_with_slowest_car_infront_sorted = sort_matrix_by_nth_and_mth_column(cars_with_slowest_car_infront, grp_idx, val_idx)
    
    return cars_with_slowest_car_infront, mask_valid_cars

def cars_index_with_track_mod(cars, track_sizes):
    mod_for_cars = track_sizes[(cars[:, 0][..., None] == track_sizes[:, 0]).nonzero(as_tuple=True)[1]]
    cars_with_indicies_mod_track = torch.clone(cars)
    cars_with_indicies_mod_track[:, 1] = torch.remainder(cars[:, 1], mod_for_cars[:, 1])[:]

    return cars_with_indicies_mod_track

def eligible_for_takeover(a, mods, val_idx=1):
    """
    - two lane case
    - all cars with their original indicies (no adjustment to put them in one lane)
    - those cars where the index occurs twice cannot change the lane

    -- outdated --
    a = torch.tensor([[0, 1, 4], [0, 2, 3], [0, 3, 5], [1, 2, 3], [1, 3, 7], [0, 4, 3], [1, 10, 1], [1, 11, 3], [0, 9, 5], [0, 10, 3]], dtype=torch.float)
    b = tensor([0, 1, 0, 1, 0, 0, 0, 1, 0, 0], dtype=torch.int32)
    c = tensor([0, 1, 1, 1, 1, 0, 0, 1, 1, 0], dtype=torch.int32)
    d = tensor([0, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=torch.int32)
    e = tensor([1, 0, 0, 1, 0, 0, 0, 1, 0, 1], dtype=torch.int32)
    """
    a_mod = torch.clone(a)
    a_mod[:, 1] = cars_index_with_track_mod(a, mods)[:, 1]
    sorted_matrix = sort_matrix_by_nth_entry(a_mod, val_idx)
    b = (sorted_matrix[:, val_idx] - torch.cat([sorted_matrix[1:, val_idx], torch.tensor([sorted_matrix[-2, val_idx]])]) == 0).int()
    c = torch.roll(b, 1) | b
    d = reverse_sort_matrix_by_nth_entry(a, c)
    e = 1 - d
    
    return e

def cars_desiring_takeover(cars, track_sizes, grp_idx = 0, val_idx = 1, speed_idx = 2):
    """
    - two lane case
    - all cars with their original indicies (no adjustment to put them in one lane)
    - respect the last car of a track when using modulo-tracks

    returns:
     - original_cars_to_overtake: positions of cars (in their original index) which try to overtake
    """
    # Sort cars (since some cars might have changed the lane)
    cars_sorted = sort_matrix_by_nth_and_mth_column(cars)
    # mask_valid_cars works for sorted arrays
    cars_with_slowest_cars_infront, mask_valid_cars = put_slowest_car_infront(cars_sorted, track_sizes)
    
    # Put all cars in one lane
    max_values_per_lane = mapping_max_values_to_prev_lanes(cars_with_slowest_cars_infront)
    cars_in_single_lane = add_to_matrix_mapping_values(cars_with_slowest_cars_infront, max_values_per_lane)

    # Fit order cars, Update the cars
    cars_update_sorted = sort_matrix_by_nth_and_mth_column(cars_in_single_lane)
    cars_update_sorted[:, 1] += cars_update_sorted[:, 2]
    
    # Based on the cars in their respective groups, which are the ones that try to overtake?
    updated_cars_to_overtake = find_values_that_must_be_reordered(cars_update_sorted[:, 1])
    cars_to_overtake = reverse_sort_matrix_by_nth_and_mth_column(cars_in_single_lane, updated_cars_to_overtake)
    no_sort_mask_valid_cars = reverse_sort_matrix_by_nth_and_mth_column(cars_in_single_lane, mask_valid_cars)
    
    original_cars_to_overtake = reverse_sort_matrix_by_nth_and_mth_column(cars, cars_to_overtake[no_sort_mask_valid_cars])
    
    return original_cars_to_overtake

def cars_that_overtake(cars, track_sizes):
    cars_that_might_overtake = cars_desiring_takeover(cars, track_sizes) & eligible_for_takeover(cars, track_sizes)
    prob_that_cars_overtake = cars_that_might_overtake * cars[:, 3]
    cars_that_overtake = (prob_that_cars_overtake > torch.rand(prob_that_cars_overtake.shape[0]))
    
    return cars_that_overtake

def update_cars(cars, track_sizes):
    """
    - two car lanes
    """
    cars_sorted = sort_matrix_by_nth_and_mth_column(cars)
    cars_that_change_lane = cars_that_overtake(cars_sorted, track_sizes)
    updated_cars = update_cars_with_lane_changes(cars, track_sizes, cars_that_change_lane)

    return updated_cars

#### Hackie
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

def sort_matrix_by_nth_entry(matrix, n=0):
    sorted_values, sorted_indicies = torch.sort(matrix[:, n], stable=True)
    
    return matrix[sorted_indicies]

def sort_matrix_by_group(matrix, grp_idx=0, val_idx=1):
    transposed = matrix.T
    sorted_indices = torch.argsort(transposed[grp_idx] * matrix.shape[grp_idx] + transposed[val_idx])
    
    return matrix[sorted_indices]

def sort_matrix_by_nth_and_mth_column(matrix, mth_col=0, nth_col=1):
    """
    First priority is nth-column, second priority is mth-column
    """

    matrix = sort_matrix_by_nth_entry(matrix, nth_col)
    matrix = sort_matrix_by_nth_entry(matrix, mth_col)

    return matrix

def reverse_sort_matrix_by_nth_entry(matrix, modified_matrix, n=0):
    sorted_values, sorted_indicies = torch.sort(matrix[:, n], stable=True)
    _, reverse_sort_indicies = torch.sort(sorted_indicies)

    return modified_matrix[reverse_sort_indicies]

def reverse_sort_matrix_by_group(matrix, modified_matrix, grp_idx=0, val_idx=1):
    transposed = matrix.T
    sorted_indices = torch.argsort(transposed[grp_idx] * matrix.shape[grp_idx] + transposed[val_idx])
    reverse_sorted_indicies, _ = torch.sort(sorted_indices)
    
    return modified_matrix[reverse_sorted_indicies]

def reverse_sort_matrix_by_nth_and_mth_column(matrix, modified_matrix, mth_col=0, nth_col=1):
    modified_matrix = reverse_sort_matrix_by_nth_entry(matrix, modified_matrix, mth_col)
    reverse_sorted_matrix = reverse_sort_matrix_by_nth_entry(matrix, modified_matrix, nth_col)

    return reverse_sorted_matrix

#### 
