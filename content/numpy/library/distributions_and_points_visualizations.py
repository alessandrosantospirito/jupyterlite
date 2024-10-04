import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from itertools import cycle

def visualize_points(pts):
    plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5)
    plt.axis('equal')
    plt.grid()

    return plt

def create_ellipse_points(mean, cov, n_std=1.96, n_points=100):
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    t = np.linspace(0, 2*np.pi, n_points)
    ellipse_x = n_std * np.sqrt(eigenvalues[0]) * np.cos(t)
    ellipse_y = n_std * np.sqrt(eigenvalues[1]) * np.sin(t)
    
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    
    ellipse_points = np.dot(np.column_stack([ellipse_x, ellipse_y]), R.T) + mean
    return ellipse_points[:, 0], ellipse_points[:, 1]

def plot_confidence_ellipses(distributions, n_std=1.96):
    fig, ax = plt.subplots()

    cmap = plt.get_cmap('viridis')
    colors = cycle(cmap(np.linspace(0, 1, len(distributions))))
    patches = []

    for idx, dist in enumerate(distributions):
        mean, cov = dist[0], np.array([dist[1], dist[2]])
        x, y = create_ellipse_points(mean, cov, n_std)
        
        color = next(colors)
        
        # Create polygon
        polygon_points = np.column_stack([x, y])
        poly = Polygon(polygon_points, closed=True, fill=True, color=color, alpha=0.5)
        ax.add_patch(poly)
        patches.append(poly)

    ax.set_title('Confidence Ellipses')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(patches, ['Distribution {}'.format(i) for i in range(len(distributions))])
    ax.axis('equal')

    return plt, ax

def plot_distributions_with_points(distributions, points, n_std=1.96):
    fig, ax = plot_confidence_ellipses(distributions, n_std)
    
    ax.plot(points[:, 0], points[:, 1], 'ko')

    return plt

def plot_distributions_with_colored_points(distributions, points, points_to_distribution, n_std=1.96):
    fig, ax = plot_confidence_ellipses(distributions, n_std)
    
    # Plot all points in a default color (black)
    ax.plot(points[:, 0], points[:, 1], 'ko', label='All Points')

    # Use the same colormap to get consistent colors
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(distributions)))

    for dist_idx, point_idx in enumerate(points_to_distribution):
        ax.plot(points[point_idx, 0], points[point_idx, 1], 'o', color=colors[dist_idx], label=f'Point to Distribution {dist_idx}')
    
    ax.legend()
    return plt, ax

def plot_distributions_with_colored_points(distributions, points, points_to_distribution, n_std=1.96):
    fig, ax = plot_confidence_ellipses(distributions, n_std)
    
    # Plot all points in a default color (black)
    ax.plot(points[:, 0], points[:, 1], 'ko', label='All Points')

    # Use the same colormap to get consistent colors
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(distributions)))

    for dist_idx, point_idx in enumerate(points_to_distribution):
        if point_idx == -1: continue
        ax.plot(points[point_idx, 0], points[point_idx, 1], 'o', color=colors[dist_idx], label=f'Point to Distribution {dist_idx}')
    
    ax.legend()
    return plt, ax

def plot_gmm_distributions(distributions, weights, ids, n_std=1.96):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    colors = cm.tab10.colors

    max_weight_index = np.argmax(weights)
    
    for dist_index, (dist, weight, id_set) in enumerate(zip(distributions, weights, ids)):
        for idx, params in zip(id_set, dist):
            # pdb.set_trace()
            mean, cov_matrix = params[0], params[1:]
            ellipse_x, ellipse_y = create_ellipse_points(mean, cov_matrix, n_std)
            
            # Use the id to select a consistent color
            color_index = idx % len(colors)
            if dist_index == max_weight_index:
                ax.fill(ellipse_x, ellipse_y, color=colors[color_index], alpha=weight, 
                        label=f'Distribution {idx}', edgecolor='black')
            else:
                ax.fill(ellipse_x, ellipse_y, color=colors[color_index], alpha=weight, edgecolor='black')
    
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('GMM Distributions with Weights as Opacity')
    plt.grid(True)
    
    return plt

# example: point-mapping: [-1, -1, -1, 5, 0, 2]
def plot_gmm_distributions_with_points(distributions, weights, ids, points, point_id_mapping, n_std=1.96):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    colors = cm.tab10.colors

    max_weight_index = np.argmax(weights)
    
    for dist_index, (dist, weight, id_set) in enumerate(zip(distributions, weights, ids)):
        for idx, params in zip(id_set, dist):
            mean, cov = np.array(params[0]), np.array(params[1:])
            cov_matrix = np.array(cov)
            ellipse_x, ellipse_y = create_ellipse_points(mean, cov_matrix, n_std)
            
            # Use the id to select a consistent color
            color_index = idx % len(colors)
            if dist_index == max_weight_index:
                ax.fill(ellipse_x, ellipse_y, color=colors[color_index], alpha=weight, 
                        label=f'Distribution {idx}', edgecolor='black')
            else:
                ax.fill(ellipse_x, ellipse_y, color=colors[color_index], alpha=weight, edgecolor='black')
    
    # Plot each point
    for point, mapping in zip(points, point_id_mapping):
        if mapping != -1:
            ax.scatter(point[0], point[1], color=colors[mapping % len(colors)], edgecolor='black')
        else:
            ax.scatter(point[0], point[1], color='black', edgecolor='black')
    
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('GMM Distributions with Weights as Opacity')
    plt.grid(True)
    
    return plt
