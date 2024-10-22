import csv
import cv2
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader

def save_params_anonymous(func):
    def wrapper(*args, **kwargs):
        prefix = "_"
        globals().update({prefix + k: v for k, v in zip(func.__code__.co_varnames, args)})
        return func(*args, **kwargs)
    return wrapper

def save_params(func):
    def wrapper(*args, **kwargs):
        globals().update({k: v for k, v in zip(func.__code__.co_varnames, args)})
        return func(*args, **kwargs)
    return wrapper

# Example for two points
def dist_angle_from_coordinates(origin_coords, target_coords):
    # center to 0-coordinate
    target_coords -= origin_coords
    
    dist = np.linalg.norm(target_coords)
    angle = np.rad2deg(np.arctan(target_coords[0][1]/target_coords[0][0]))

    return dist, angle

def get_classnames_boxes_from_csv(csv_file_name, iteration):
    classes_names, boxes = [], []

    with open(csv_file_name, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['iteration']) == iteration:
                classes_names.append(row['class'])
                boxes.append({
                    'top': float(row['top']),
                    'left': float(row['left']),
                    'bottom': float(row['bottom']),
                    'right': float(row['right']),
                    'center_x': float(row['center_x']),
                    'center_y': float(row['center_y'])
                })
            if int(row['iteration']) > iteration:
                return classes_names, boxes
    return classes_names, boxes

from itertools import product

def cartesian_product_for_nodes(list_nodes, self_edge=True):
    all_pairs = list(product(list_nodes, repeat=2))

    if self_edge: return np.array(all_pairs)
        
    return np.array([pair for pair in all_pairs if pair[0] != pair[1]])

def dist_angle_from_matrix(matrix, rules):
    val_edge_pairs = np.array([matrix[b] - matrix[a] for [a,b] in rules])
    
    return np.linalg.norm(val_edge_pairs, axis=1), np.rad2deg(np.arctan(val_edge_pairs[:, 1] / val_edge_pairs[:, 0]))
