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
    a = tensor([-82, -83, -88, -92, -86, -91, -89, -94, -89, -93, -93, -95, -96, -100, -96, -104, -103, -98, -100, -100, -100, -109, -104, -110, -105, -114, -89, -85, -106, -58])
    b = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    c = tensor([-83, -85, -91, -96, -91, -97, -96, -102, -98, -103, -104, -107, -109, -114, -111, -120, -120, -116, -119, -120, -121, -131, -127, -134, -130, -140, -116, -113, -135, -88])
    d = tensor([-140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -140, -135, -135, -135, -88])
    e = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1], dtype=torch.int32)
    """
    b = torch.arange(1, a.shape[0] + 1)
    c = a - b
    d, _ = torch.cummin(c.flip(0), dim=0)
    d = d.flip(0)
    e = (c == d).int()

    return 1 - e

def adjust_cars_no_lane_change(a):
    """
    a = tensor([ 3, 10,  8, 11,  5, 13,  7, 14, 18, 15, 19, 20, 13, 21, 16, 24, 19, 24, 23, 22, 28, 22, 27, 26, 33, 32, 30, 32, 31, 36])
    b = tensor([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1], dtype=torch.int32)
    c = tensor([4, 3, 2, 1, 0, 1, 0, 5, 4, 3, 2, 1, 0, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 0, 4, 3, 2, 1, 0, 0])
    d = tensor([ 0,  0,  0,  0,  5,  0,  7,  0,  0,  0,  0,  0, 13,  0,  0,  0,  0,  0, 0,  0,  0, 22,  0, 26,  0,  0,  0,  0, 31, 36])
    e = tensor([ 5,  0,  0,  0,  0,  7,  0, 13,  0,  0,  0,  0,  0, 22,  0,  0,  0,  0, 0,  0,  0,  0, 26,  0, 31,  0,  0,  0,  0, 36])
    f = tensor([ 5,  5,  5,  5,  5,  7,  7, 13, 13, 13, 13, 13, 13, 22, 22, 22, 22, 22, 22, 22, 22, 22, 26, 26, 31, 31, 31, 31, 31, 36])
    g = tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 36])
    """
    b = 1 - find_values_that_must_be_reordered(a)
    c = fill_zero_values_descending_rest_with_zero(b)
    d = a * b
    e = shift_values_to_next_available_zero(d)
    f = fill_zero_places(e)
    g = f - c

    return g
