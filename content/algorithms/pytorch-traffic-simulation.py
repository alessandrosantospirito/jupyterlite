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

def eligible_for_takeover(a, val_idx=1):
    """
    - two lane case
    - all cars with their original indicies (no adjustment to put them in one lane)
    - those cars where the index occurs twice cannot change the lane

    a = torch.tensor([[0, 1, 4], [0, 2, 3], [0, 3, 5], [1, 2, 3], [1, 3, 7], [0, 4, 3], [1, 10, 1], [1, 11, 3], [0, 9, 5], [0, 10, 3]], dtype=torch.float)
    b = tensor([0, 1, 0, 1, 0, 0, 0, 1, 0, 0], dtype=torch.int32)
    c = tensor([0, 1, 1, 1, 1, 0, 0, 1, 1, 0], dtype=torch.int32)
    d = tensor([0, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=torch.int32)
    e = tensor([1, 0, 0, 1, 0, 0, 0, 1, 0, 1], dtype=torch.int32)
    """

    sorted_matrix = sort_matrix_by_nth_entry(a, val_idx)
    b = (sorted_matrix[:, val_idx] - torch.cat([sorted_matrix[1:, val_idx], torch.tensor([sorted_matrix[-2, val_idx]])]) == 0).int()
    c = torch.roll(b, 1) | b
    d = reverse_sort_matrix_by_nth_entry(a, c)
    e = 1 - d
    
    return e

def mapping_max_values_to_prev_lanes(cars):
    """
    - two lanes
    """
    max_values_per_lane = maximum_value_by_grp(cars, 0, 1)
    mapping_max_values_per_lane = torch.clone(max_values_per_lane)
    mapping_max_values_per_lane[1:, 1] = mapping_max_values_per_lane[:-1, 1]
    mapping_max_values_per_lane[0, 1] = 0

    return mapping_max_values_per_lane

def cars_desiring_takeover(cars, grp_idx = 0, val_idx = 1, speed_idx = 2):
    """
    - two lane case
    - all cars with their original indicies (no adjustment to put them in one lane)

    returns:
     - original_cars_to_overtake: positions of cars (in their original index) which try to overtake
    """

    # before the updates, no car is eligible for takeover (all cars are already in order)
    mapping_max_values_per_lane = mapping_max_values_to_prev_lanes(cars)
    cars_in_single_lane = add_to_matrix_mapping_values(cars, mapping_max_values_per_lane)
    
    # Update the cars
    cars_update = torch.clone(cars_in_single_lane)
    cars_update[:, val_idx] += cars_update[:, speed_idx]
    
    # Adjust the cars, such that they could fit into one lane (to execute adjust_cars_no_lane_change)
    cars_update_ordered_by_grp = sort_matrix_by_nth_entry(cars_update)

    # Based on the cars in their respective groups, which are the ones that try to overtake?
    updated_cars_to_overtake = find_values_that_must_be_reordered(cars_update_ordered_by_grp[:, 1])
    original_cars_to_overtake = reverse_sort_matrix_by_nth_entry(cars_update, updated_cars_to_overtake)

    return original_cars_to_overtake

def cars_that_overtake(cars):
    cars_that_might_overtake = cars_desiring_takeover(cars) & eligible_for_takeover(cars)
    prob_that_cars_overtake = cars_that_might_overtake * cars[:, 3]
    cars_that_overtake = (prob_that_cars_overtake > torch.rand(prob_that_cars_overtake.shape[0]))

    return cars_that_overtake

def update_cars(cars):
    """
    - two car lanes
    """
    
    cars_that_change_lane = cars_that_overtake(cars)
    cars[:, 0] += cars_that_change_lane
    cars[:, 0] = torch.remainder(cars[:, 0], 2)

    # Sort cars (which might have updated
    cars_sorted = sort_matrix_by_nth_and_mth_column(cars)
    
    # Put all cars in one lane
    max_values_per_lane = mapping_max_values_to_prev_lanes(cars_sorted)
    cars_in_single_lane = add_to_matrix_mapping_values(cars, max_values_per_lane)

    # Update the cars
    cars_update = torch.clone(cars_in_single_lane)
    cars_update[:, 1] += cars_update[:, 2]

    # Adjust the cars, such that they could fit into one lane (to execute adjust_cars_no_lane_change)
    cars_update_in_lane = torch.clone(cars_update)
    cars_update_in_lane[:, 1] = adjust_cars_no_lane_change(cars_update[:, 1])[:]

    # Remove the lane offset and put the cars to the original index
    neg_max_values_per_lane = torch.clone(max_values_per_lane)
    neg_max_values_per_lane[:, 1] = - neg_max_values_per_lane[:, 1]
    cars_update_multi_lane = add_to_matrix_mapping_values(cars_update_in_lane, neg_max_values_per_lane)

    return cars_update_multi_lane
