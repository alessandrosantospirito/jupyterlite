import torch

def fill_zero_places(a):
    a = torch.cat([torch.tensor([-1], device=a.device), a])
    nonzero_indices = torch.nonzero(a).squeeze()
    diff = torch.diff(torch.cat([nonzero_indices, torch.tensor([a.shape[0]], device=a.device)]))
    b = a[nonzero_indices].repeat_interleave(diff)
    b[b < 0] = 0
    c = b[1:]
    return c

def shift_values_to_next_available_zero(a):
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
    b = (a > 0).int()
    cumsum = torch.cumsum(1 - b, dim=0)
    c = (cumsum * (1 - b) - torch.maximum(b * cumsum, torch.zeros_like(cumsum)).cummax(dim=0)[0]) * (1 - b)
    
    return c

def find_one_zero_pattern(a):
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

def find_values_that_must_be_reordered(a):
    b = torch.arange(1, a.shape[0] + 1)
    c = a - b
    d, _ = torch.cummin(c.flip(0), dim=0)
    d = d.flip(0)
    e = (c == d).int()

    return 1 - e

def adjust_cars_no_lane_change(a):
    b = 1 - find_values_that_must_be_reordered(a)
    c = fill_zero_values_descending_rest_with_zero(b)
    d = a * b
    e = shift_values_to_next_available_zero(d)
    f = fill_zero_places(e)
    g = f - c

    return g
