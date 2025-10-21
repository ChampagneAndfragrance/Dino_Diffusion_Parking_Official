import numpy as np
import matplotlib.pyplot as plt

def deg2rad_trajectory_2D(traj):
    traj = traj.cpu().numpy()
    traj[:, 2] = np.deg2rad(traj[:, 2])
    return traj

def plot_trajectory_with_yaw(traj, arrow_length=0.5, invert_x=False, invert_y=False):
    """
    traj: torch.Tensor of shape (T, 3) where columns = (x, y, rad)
    arrow_length: length of yaw arrows in plot units
    """
    x = traj[:, 0]
    y = traj[:, 1]
    yaw = traj[:, 2]  # in radians

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='o', markersize=3, label="Trajectory")
    plt.scatter(x[0], y[0], color='green', label="Start", zorder=5)
    plt.scatter(x[-1], y[-1], color='red', label="End", zorder=5)

    # Compute arrow components
    dx = np.cos(yaw) * arrow_length
    dy = np.sin(yaw) * arrow_length

    # Draw arrows
    plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)

    plt.axis('equal')
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Trajectory with Yaw")
    plt.legend()
    plt.grid(True)
    if invert_x:
        plt.gca().invert_xaxis()
    if invert_y:
        plt.gca().invert_yaxis()
    plt.show()

def invert_trajectory_2D(traj):
    """
    Invert a trajectory of poses.
    
    Args:
        traj: np.ndarray of shape (T, 3), where each row is [x, y, theta]

    Returns:
        np.ndarray of shape (T, 3), inverse poses
    """
    x, y, theta = traj[:, 0], traj[:, 1], traj[:, 2]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_inv = -x * cos_theta - y * sin_theta
    y_inv =  x * sin_theta - y * cos_theta
    theta_inv = -theta

    return np.stack([x_inv, y_inv, theta_inv], axis=1)
