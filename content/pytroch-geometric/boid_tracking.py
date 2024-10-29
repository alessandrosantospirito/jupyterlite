import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import sys
import os
from IPython.display import display, HTML

## Calculations
def velocity_vector_rad(velocities, edge_connections):
    vector_angles = np.arctan2(velocities[:, 0], (velocities[:, 1]))
    vector_rad = np.mod([vector_angles[b] - vector_angles[a] for [a, b] in edge_connections], np.pi)

    return vector_rad

def loss_for_dist_degree_angle_pairs(true_attr, pred_attr, epsilon=1e-8):
    loss_dist = F.mse_loss(true_attr[:, 0], pred_attr[:, 0])
    # https://stats.stackexchange.com/questions/425234/loss-function-and-encoding-for-angles
    loss_angle = torch.mean(torch.sqrt(1 - torch.cos(pred_attr[:, 1] - true_attr[:, 1]) + epsilon))

    return loss_dist + loss_angle

def calc_point_based_on_coord_and_edges(flocks_pos, flocks_edge_attr):
    dx = flocks_edge_attr[:, 0] * torch.cos(flocks_edge_attr[:, 1])
    dy = flocks_edge_attr[:, 0] * torch.sin(flocks_edge_attr[:, 1])

    return torch.column_stack([dx + flocks_pos[:, 0], dy + flocks_pos[:, 1]])

## Visualizations
def visualize_boids(list_of_birds_pos, list_of_birds_vel, boundaries):
    list_of_birds_pos = [list_of_birds_pos[i].cpu().numpy() for i in range(len(list_of_birds_pos))]
    list_of_birds_vel = [list_of_birds_vel[i].cpu().numpy() for i in range(len(list_of_birds_vel))]
    
    offset = 0
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(boundaries[0], boundaries[1])
    ax.set_ylim(boundaries[0], boundaries[1])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    initial_positions = list_of_birds_pos[0]
    initial_velocities = list_of_birds_vel[0]
    
    scatter = ax.scatter([], [], c='orange', s=10)
    
    quiver = ax.quiver(initial_positions[:, 0], initial_positions[:, 1], 
                       initial_velocities[:, 0], initial_velocities[:, 1], 
                       angles='xy', scale_units='xy', scale=1, color='blue')
    
    def update(frame):
        positions = list_of_birds_pos[frame]
        velocities = list_of_birds_vel[frame]
    
        scatter.set_offsets(positions)
    
        quiver.set_offsets(positions)
        quiver.set_UVC(velocities[:, 0], velocities[:, 1])
    
        return scatter, quiver
    
    ani = FuncAnimation(fig, update, frames=len(list_of_birds_pos), interval=50, blit=True)
    plt.close(fig)
    
    return HTML(ani.to_html5_video())

def plot_observations_and_velocities_to_grid(observation, velocities, box_top):
    # plt.close('all')
    
    offset = 0
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(offset, box_top - offset)
    ax.set_ylim(offset, box_top - offset)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    scatter = ax.scatter([], [], c='orange', s=10)
    
    quiver = ax.quiver(velocities[:, 0], velocities[:, 1],
                       velocities[:, 0], velocities[:, 1],
                       angles='xy', scale_units='xy', scale=1, color='blue')
    

    scatter.set_offsets(observation)
    quiver.set_offsets(observation)

    plt.show()
    plt.close(fig)

def plot_observations_to_grid(observation, box_top):
    fig, ax = plt.subplots()
    ax.set_xlim(0, box_top)
    ax.set_ylim(0, box_top)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    scatter = ax.scatter([], [], c='orange', s=10)
    scatter.set_offsets(observation.cpu().numpy()[:])

    plt.show()
    plt.close(fig)

def visualize_vectorfield_and_interp_vectorfield(X, Y, U, S, V, XX, YY, UU, SS, VV):
    div = lambda n, d: np.divide(n, d, out = np.zeros_like(d), where=d!=0)
    
    fig, ax = plt.subplots(1, 1)
    quiver = ax.quiver(X, Y, div(U, S), div(V, S), S, cmap='autumn')
    ax.set_aspect('equal')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    fig.colorbar(quiver)
    plt.show()
    plt.close()
    
    # Visualize interpolated field
    fig, ax = plt.subplots(1,1)
    stream = ax.streamplot(XX.T, YY.T, div(UU, SS).T, div(VV, SS).T, color=SS.T, density=1, cmap='autumn')
    fig.colorbar(stream.lines)
    ax.set_aspect('equal')
    plt.show()
    plt.close()

def plot_histogram(hist, box_top):
    fig, ax = plt.subplots()
    hist_normalized = hist / hist.max()

    cmap = plt.get_cmap('viridis')
    im = ax.imshow(hist_normalized.cpu().numpy(), cmap=cmap, extent=[0, box_top, 0, box_top], origin='lower', aspect='equal')
    
    # cbar = plt.colorbar(im)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.show()
    plt.close(fig)

def plot_histogram_as_point_plot(histogram, box_top):
    fig, ax = plt.subplots()
    ax.set_xlim(0, box_top)
    ax.set_ylim(0, box_top)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    grid_size = histogram.shape[0]
    y_indices, x_indices = torch.nonzero(histogram, as_tuple=True)
    x_coords = x_indices.float() * (box_top / grid_size)
    y_coords = y_indices.float() * (box_top / grid_size)
    
    points = torch.stack((x_coords, y_coords), dim=1)
    scatter = ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), 
                         c='orange', s=10)

    plt.show()
    plt.close(fig)

def visualize_boids_as_histograms(list_of_birds_pos, grid_size, box_top):
    # plt.close('all')
    
    offset = 0
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(offset, box_top - offset)
    ax.set_ylim(offset, box_top - offset)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    scatter = ax.scatter([], [], c='orange', s=10)

    def update(frame):
        histogram = positions_to_grid(list_of_birds_pos[frame], grid_size, box_top)

        y_indices, x_indices = torch.nonzero(histogram, as_tuple=True)
        x_coords = x_indices.float() * (box_top / grid_size)
        y_coords = y_indices.float() * (box_top / grid_size)
    
        points = torch.stack((x_coords, y_coords), dim=1)
        scatter.set_offsets(points.cpu().numpy())
        return scatter,
    
    ani = FuncAnimation(fig, update, frames=len(list_of_birds_pos), interval=50, blit=True)
    plt.close(fig)
    
    return HTML(ani.to_html5_video())

def visualize_boids_predictions(list_of_birds_pos, list_ground_truth_pos, box_top):
    # plt.close('all')
    
    list_of_birds_pos = [list_of_birds_pos[i].cpu().numpy() for i in range(len(list_of_birds_pos))]
    list_ground_truth_pos = [list_ground_truth_pos[i].cpu().numpy() for i in range(len(list_ground_truth_pos))]
    
    offset = 0
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(offset, box_top - offset)
    ax.set_ylim(offset, box_top - offset)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    initial_positions = list_of_birds_pos[0]
    
    scatter_pred = ax.scatter([], [], c='orange', s=30, label='Predictions')
    scatter_truth = ax.scatter([], [], c='blue', s=30, label='Ground Truth')
    
    ax.legend(loc='upper right')
    
    def update(frame):
        positions_pred = list_of_birds_pos[frame]
        positions_truth = list_ground_truth_pos[frame]
        
        scatter_pred.set_offsets(positions_pred)
        scatter_truth.set_offsets(positions_truth)
        
        return scatter_pred, scatter_truth
    
    ani = FuncAnimation(fig, update, frames=len(list_of_birds_pos), interval=50, blit=True)
    plt.close(fig)
    
    return HTML(ani.to_html5_video())