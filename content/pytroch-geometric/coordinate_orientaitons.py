from itertools import product
import numpy as np

def dist_angle_from_coordinates(origin_coords, target_coords):
    # center to 0-coordinate
    target_coords -= origin_coords
    
    dist = np.linalg.norm(target_coords)
    angle = np.rad2deg(np.arctan(target_coords[0][1]/target_coords[0][0]))

    return dist, angle

def cartesian_product_for_nodes(list_nodes, self_edge=True):
    all_pairs = list(product(list_nodes, repeat=2))

    if self_edge: return np.array(all_pairs)
        
    return np.array([pair for pair in all_pairs if pair[0] != pair[1]])

def dist_angle_from_matrix(matrix, rules):
    val_edge_pairs = np.array([matrix[b] - matrix[a] for [a,b] in rules])
    
    return np.linalg.norm(val_edge_pairs, axis=1), np.rad2deg(np.arctan2(val_edge_pairs[:, 1], (val_edge_pairs[:, 0])))
