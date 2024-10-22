import numpy as np

def calculate_rotated_line(start_point, end_point, rotation_angle):
    vector = np.array(end_point) - np.array(start_point)
    length = np.linalg.norm(vector)
    assert np.isclose(length, 1), f"Line length is not 1, it is: {length}. Input: {start_point}, {end_point}, {rotation_angle}"

    rotation_matrix = np.array([[vector[0], -vector[1]], [vector[1], vector[0]]])
    rotated_vector = rotation_matrix @ np.array([[1 + np.cos(rotation_angle)], [np.sin(rotation_angle)]])
    new_end_point = np.array(start_point).reshape(2, 1) + rotated_vector

    # Normalize the new line to ensure its length is 1
    new_vector = new_end_point.flatten() - end_point
    new_end_point = end_point + new_vector / np.linalg.norm(new_vector)

    return new_end_point.flatten()

def generate_point_sequence(point_sequence, rotation_angles):
    for angle in rotation_angles:
        start_point, end_point = point_sequence[-2], point_sequence[-1]
        point_sequence.append(calculate_rotated_line(start_point, end_point, angle))
        
    return np.array(point_sequence)

path = "/workspaces/jupyterlite/content/pytorch-physics/html/01-angle_with_vector-space_transformation.html"
with open(path, 'r', encoding='utf-8') as file:
    calculate_rotated_line.__html__ = file.read()
