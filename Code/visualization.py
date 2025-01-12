import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

def makeplot_animation_only_agents(positions, velocities, N, T, h, Z, SAVEFIG=False, filename_template="animation_only_agents"):
    """
    Create an animated plot of particle movements with quiver arrows indicating velocities.

    Parameters:
    - positions: Array of shape (T, N, 2) containing positions of N agents over T timesteps.
    - velocities: Array of shape (T, N, 2) containing velocities of N agents over T timesteps.
    - N: Number of agents.
    - T: Total number of frames (timesteps) for the animation.
    - h: Time step size.
    - Z: Desired relative spatial configuration between agents (N x 2 array). 
    - SAVEFIG: Boolean. If True, saves the animation to a file.
    - filename_template: Template for saving filenames if SAVEFIG=True.
    """
    fig, ax = plt.subplots(figsize=(7, 6)) 
    ax.set_aspect('equal')

    # Determine axis limits based on Z (desired spatial configuration)
    x_min, x_max = np.min(Z[:, 0]), np.max(Z[:, 0])
    y_min, y_max = np.min(Z[:, 1]), np.max(Z[:, 1])
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_xlabel('X')
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_ylabel('Y')

    # Particle properties
    dotsize = 600 / N
    dot_colors = 'blue'  # All particles in blue

    # Initial scatter plot and quiver plot
    scat = ax.scatter(
        positions[0, :, 0], positions[0, :, 1],
        s=dotsize, c=dot_colors, alpha=0.6
    )

    # Normalize velocities for quiver plot (can be removed to obtain long arrows)
    velocities_normalized = velocities / np.linalg.norm(velocities, axis=2, keepdims=True)
    velocities_normalized[np.isnan(velocities_normalized)] = 0  # Handle cases where velocities are zero

    qax = ax.quiver(
        positions[0, :, 0], positions[0, :, 1],
        velocities_normalized[0, :, 0], velocities_normalized[0, :, 1],
        angles='xy', width=0.001*(30/N), scale=70*(N/30), color='black'  # Arrows in black
    )

    def _update_plot(frame):
        """
        Update function for the animation.

        Parameters:
        - frame: Current frame index.

        Returns:
        - Updated scatter and quiver plots.
        """
        time = frame * h
        ax.set_title(f'Agents movements at time: {time:.3f}s', fontsize=12)

        # Update scatter plot positions
        scat.set_offsets(positions[frame])

        # Update quiver plot positions and directions
        qax.set_offsets(positions[frame])
        qax.set_UVC(velocities_normalized[frame, :, 0], velocities_normalized[frame, :, 1])

        return scat, qax

    # Create the animation
    ani = FuncAnimation(fig, _update_plot, frames=T+1, interval=200, blit=False)  # Increased interval for slower updates
    # NOTE : With blit=True, only elements explicitly returned by _update_plot are redrawn, and titles/labels are ignored. This prevents the title from being updated.

    # Save the animation if requested
    if SAVEFIG:
        mp4_filename = f"{filename_template}.mp4"

        # Save as MP4 using ffmpeg
        print(f"Saving animation as {mp4_filename}...")
        ani.save(mp4_filename, dpi=300, fps=30, writer='ffmpeg')
        print(f"Animation saved as {mp4_filename}")

    # Always show the animation
    plt.show()

def make_plot_snapshot_only_agents(positions, velocities, N, times, h, Z, SAVEFIG=False, filename_template="snapshot_only_agents"):
    """
    Create static plots of particle movements at specified times with quiver arrows indicating velocities.

    Parameters:
    - positions: Array of shape (T, N, 2) containing positions of N agents over T timesteps.
    - velocities: Array of shape (T, N, 2) containing velocities of N agents over T timesteps.
    - N: Number of agents.
    - times: List of specific timesteps at which to generate the plots.
    - h: Time step size.
    - Z: Desired relative spatial configuration between agents (N x 2 array). 
    - SAVEFIG: Boolean. If True, saves each plot to a file.
    - filename_template: Template for saving filenames if SAVEFIG=True.
    """
    # Determine axis limits based on Z (desired spatial configuration)
    x_min, x_max = np.min(Z[:, 0]), np.max(Z[:, 0])
    y_min, y_max = np.min(Z[:, 1]), np.max(Z[:, 1])
    margin = 0.1 * max(x_max - x_min, y_max - y_min)

    # Particle properties
    dotsize = 600 / N
    dot_colors = 'blue'  # All particles in blue

    # Normalize velocities for quiver plot 
    velocities_normalized = velocities / np.linalg.norm(velocities, axis=2, keepdims=True)
    velocities_normalized[np.isnan(velocities_normalized)] = 0  # Handle cases where velocities are zero

    for time_step in times:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_aspect('equal')
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_xlabel('X')
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_ylabel('Y')

        # Get current positions and velocities for the time step
        current_positions = positions[time_step]
        current_velocities = velocities_normalized[time_step]

        # Scatter plot
        ax.scatter(
            current_positions[:, 0], current_positions[:, 1],
            s=dotsize, c=dot_colors, alpha=0.6
        )

        # Quiver plot
        ax.quiver(
            current_positions[:, 0], current_positions[:, 1],
            current_velocities[:, 0], current_velocities[:, 1],
            angles='xy', width=0.001*(30/N), scale=70*(N/30), color='black'
        )

        # Add title
        time = time_step * h
        ax.set_title(f'Agents movements at time: {time:.3f}s', fontsize=12)

        # Save the plot if requested
        if SAVEFIG:
            filename = f"{filename_template}_{time}.pdf"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Snapshot saved as {filename}")

        # Display the plot
        plt.show()

def compute_kinetic_energy(velocities):
    """
    Compute the kinetic energy over time.

    Parameters:
    - velocities: Array of shape (T, N, 2) containing velocities of N agents over T timesteps.

    Returns:
    - Array of kinetic energy at each timestep.
    """
    T = velocities.shape[0]
    kinetic_energy = np.zeros(T)
    for t in range(T):
        norm_velocity = np.sqrt(np.power(velocities[t, :, 0], 2) + np.power(velocities[t, :, 1], 2))
        norm_velocity=np.nan_to_num(norm_velocity)
        kinetic_energy[t] = np.sum(np.power(norm_velocity, 2))
    return kinetic_energy / 2  # Assuming mass = 1 for each agent

def compute_potential_energy(positions, adjacency_phi, beta, Z, M):
    """
    Compute the potential energy over time.

    Parameters:
    - positions: Array of shape (T, N, 2) containing positions of N agents over T timesteps.
    - adjacency_phi: Adjacency matrix indicating interaction between agents (N x N).
    - beta: Parameter for the potential energy function.
    - Z: Array of shape (N, 2) containing fixed positions to account for in potential calculations.
    - M: Coefficient for formation control.

    Returns:
    - Array of potential energy at each timestep.
    """
    T, N, _ = positions.shape
    potential_energy = np.zeros(T)

    def integral_computation(distance, beta):
        """
        Compute the integral term for potential energy calculation.

        Parameters:
        - distance: Array of distances between interacting agents.
        - beta: Parameter for the potential energy function.

        Returns:
        - Array of integral values for each distance.        
        """
        #return (np.power(1 + distance, 1 - beta) - 1) / (1 - beta) + distance
        return (np.power(distance+1,beta)-distance-1)/((beta-1)*np.power(distance+1,beta))

    for t in range(T):
        for i in range(N):
            J_phi = adjacency_phi[i, :] == 1
            distance = (np.power(positions[t, i, 0] - Z[i, 0] - positions[t, :, 0] + Z[:, 0], 2) + np.power(positions[t, i, 1] - Z[i, 1] - positions[t, :, 1] + Z[:, 1], 2))[J_phi]

            potential_energy[t] += np.sum(integral_computation(distance, beta))

        potential_energy[t] *= M/2 # Multiply by M/2 as per the formula

    return potential_energy

def make_plot_snapshot_energy(positions, velocities, adjacency_phi, beta, h, Z, M, SAVEFIG=False, filename_template="snapshot_energy"):
    """
    Plot kinetic, potential, and total energy over time.

    Parameters:
    - positions: Array of shape (T, N, 2) containing positions of N agents over T timesteps.
    - velocities: Array of shape (T, N, 2) containing velocities of N agents over T timesteps.
    - adjacency_phi: Adjacency matrix indicating interaction between agents (N x N).
    - beta: Parameter for the potential energy function.
    - h: Time step size.
    - Z: Array of shape (N, 2) containing fixed positions to account for in potential calculations.
    - M: Coefficient for formation control.
    - SAVEFIG: Boolean. If True, saves each plot to a file.
    - filename_template: Template for saving filenames if SAVEFIG=True.
    """
    T = positions.shape[0]
    time = np.arange(T) * h

    # Compute energies
    kinetic_energy = compute_kinetic_energy(velocities)
    potential_energy = compute_potential_energy(positions, adjacency_phi, beta, Z,M)
    total_energy = kinetic_energy + potential_energy

    # Plot energies
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(time, kinetic_energy, label='Kinetic energy', color='blue')
    ax.plot(time, potential_energy, label='Potential energy', color='orange')
    ax.plot(time, total_energy, label='Total energy', color='green')

    ax.set_title('Energy evolution', fontsize=12)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    # Save the plot if requested
    if SAVEFIG:
        filename = f"{filename_template}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Snapshot saved as {filename}")

    # Display the plot
    plt.show()

def make_plot_snapshot_agents_energy(positions, velocities, adjacency_phi, beta, h, Z, M, times, SAVEFIG=False, filename_template="snapshot_agents_energy"):
    """
    Create static plots of agent movements with quiver arrows and energy values at specified times.

    Parameters:
    - positions: Array of shape (T, N, 2) containing positions of N agents over T timesteps.
    - velocities: Array of shape (T, N, 2) containing velocities of N agents over T timesteps.
    - adjacency_phi: Adjacency matrix indicating interaction between agents (N x N).
    - beta: Parameter for the potential energy function.
    - h: Time step size.
    - Z: Array of shape (N, 2) containing fixed positions to account for in potential calculations.
    - M: Coefficient for formation control.
    - times: List of specific timesteps at which to generate the plots.
    - SAVEFIG: Boolean. If True, saves each plot to a file.
    - filename_template: Template for saving filenames if SAVEFIG=True.
    """
    T, N, _ = positions.shape

    # Compute energies
    kinetic_energy = compute_kinetic_energy(velocities)
    potential_energy = compute_potential_energy(positions, adjacency_phi, beta, Z, M)
    total_energy = kinetic_energy + potential_energy

    # Determine axis limits based on Z (desired spatial configuration)
    x_min, x_max = np.min(Z[:, 0]), np.max(Z[:, 0])
    y_min, y_max = np.min(Z[:, 1]), np.max(Z[:, 1])
    margin = 0.1 * max(x_max - x_min, y_max - y_min)

    # Particle properties
    dotsize = 600 / N
    dot_colors = 'blue'  # All particles in blue

    # Normalize velocities for quiver plot 
    velocities_normalized = velocities / np.linalg.norm(velocities, axis=2, keepdims=True)
    velocities_normalized[np.isnan(velocities_normalized)] = 0  # Handle cases where velocities are zero

    for time_step in times:
        fig, (scatter_ax, energy_ax) = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot configuration
        scatter_ax.set_aspect('equal')
        scatter_ax.set_xlim(x_min - margin, x_max + margin)
        scatter_ax.set_ylim(y_min - margin, y_max + margin)
        scatter_ax.set_title('Agent Movements', fontsize=12)
        scatter_ax.set_xlabel('X')
        scatter_ax.set_ylabel('Y')

        # Current positions and velocities
        current_positions = positions[time_step]
        current_velocities = velocities_normalized[time_step]

        # Scatter and quiver plots
        scatter_ax.scatter(
            current_positions[:, 0], current_positions[:, 1],
            s=dotsize, c=dot_colors, alpha=0.6
        )
        scatter_ax.quiver(
            current_positions[:, 0], current_positions[:, 1],
            current_velocities[:, 0], current_velocities[:, 1],
            angles='xy', width=0.001*(30/N), scale=70*(N/30), color='black'
        )

        # Energy plot configuration
        time = np.arange(T) * h
        energy_ax.plot(time, kinetic_energy, label='Kinetic energy', color='blue')
        energy_ax.plot(time, potential_energy, label='Potential energy', color='orange')
        energy_ax.plot(time, total_energy, label='Total energy', color='green')
        energy_ax.set_title('Energy evolution', fontsize=12)
        energy_ax.set_xlabel('Time (s)')
        energy_ax.set_ylabel('Energy')
        energy_ax.legend(loc='best')
        energy_ax.grid(True)

        # Highlight the energy at the current time step
        current_time = time_step * h
        energy_ax.plot(
            current_time, total_energy[time_step], 'ro',
            label=f'Time {current_time:.2f}s'
        )

        # Update titles with the current time
        scatter_ax.set_title(f'Agents movements at time: {current_time:.3f}s', fontsize=12)

        # Adjust layout
        plt.tight_layout()

        # Save the plot if requested
        if SAVEFIG:
            filename = f"{filename_template}_{current_time}.pdf"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Snapshot saved as {filename}")

        # Display the plot
        plt.show()

def compute_minimum_distance(positions):
    """
    Compute the minimum distance between any two agents at each timestep.

    Parameters:
    - positions: Array of shape (T, N, 2) containing positions of N agents over T timesteps.

    Returns:
    - Array of minimum distances at each timestep.
    """
    T, N, _ = positions.shape
    min_distances = np.zeros(T)

    for t in range(T):
        dist_matrix = np.sqrt(
            np.sum((positions[t, :, np.newaxis, :] - positions[t, np.newaxis, :, :]) ** 2, axis=2)
        )
        np.fill_diagonal(dist_matrix, np.inf)  # Exclude self-distances
        min_distances[t] = np.min(dist_matrix)

    return min_distances

def make_plot_minimum_distance(positions, h, SAVEFIG=False, filename_template="plot_minimum_distance"):
    """
    Plot the minimum distance between agents over time.

    Parameters:
    - positions: Array of shape (T, N, 2) containing positions of N agents over T timesteps.
    - h: Time step size.
    - SAVEFIG: Boolean. If True, saves each plot to a file.
    - filename_template: Template for saving filenames if SAVEFIG=True.
    """
    T = positions.shape[0]
    time = np.arange(T) * h
    min_distances = compute_minimum_distance(positions)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(time, min_distances, label='Minimum distance', color='purple')

    ax.set_title('Minimum distance evolution', fontsize=12)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    # Save the plot if requested
    if SAVEFIG:
        filename = f"{filename_template}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Snapshot saved as {filename}")

    # Display the plot
    plt.show()

def compute_maximal_velocity_difference(velocities):
    """
    Compute the maximal velocity difference between any two agents at each timestep.

    Parameters:
    - velocities: Array of shape (T, N, 2) containing velocities of N agents over T timesteps.

    Returns:
    - Array of maximal velocity differences at each timestep.
    """
    T, N, _ = velocities.shape
    max_velocity_diff = np.zeros(T)

    for t in range(T):
        velocity_diff_matrix = np.sqrt(
            np.sum((velocities[t, :, np.newaxis, :] - velocities[t, np.newaxis, :, :]) ** 2, axis=2)
        )
        max_velocity_diff[t] = np.max(velocity_diff_matrix)

    return max_velocity_diff

def make_plot_maximal_velocity_difference(velocities, h, SAVEFIG=False, filename_template="plot_maximal_velocity_difference"):
    """
    Plot the maximal velocity difference between agents over time.

    Parameters:
    - velocities: Array of shape (T, N, 2) containing velocities of N agents over T timesteps.
    - h: Time step size.
    - SAVEFIG: Boolean. If True, saves each plot to a file.
    - filename_template: Template for saving filenames if SAVEFIG=True.
    """
    T = velocities.shape[0]
    time = np.arange(T) * h
    max_velocity_diff = compute_maximal_velocity_difference(velocities)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(time, max_velocity_diff, label='Maximal velocity difference', color='gray')

    ax.set_title('Maximal velocity difference evolution', fontsize=12)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity difference')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    # Save the plot if requested
    if SAVEFIG:
        filename = f"{filename_template}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Snapshot saved as {filename}")

    # Display the plot
    plt.show()

def make_plot_animation_agents_energy(positions, velocities, adjacency_phi, beta, h, Z, M, SAVEFIG=False, filename_template="animation_agents_energy"):
    """
    Create an animated plot showing agent movements alongside the energy evolution.

    Parameters:
    - positions: Array of shape (T, N, 2) containing positions of N agents over T timesteps.
    - velocities: Array of shape (T, N, 2) containing velocities of N agents over T timesteps.
    - adjacency_phi: Adjacency matrix indicating interaction between agents (N x N).
    - beta: Parameter for the potential energy function.
    - h: Time step size.
    - Z: Array of shape (N, 2) containing fixed positions to account for in potential calculations.
    - M: Coefficient for formation control.
    - SAVEFIG: Boolean. If True, saves the animation to a file.
    - filename_template: Template for saving filenames if SAVEFIG=True.

    Returns:
    - An interactive matplotlib animation.
    """
    T, N, _ = positions.shape
    time = np.arange(T) * h

    # Compute energies
    kinetic_energy = compute_kinetic_energy(velocities)
    potential_energy = compute_potential_energy(positions, adjacency_phi, beta, Z, M)
    total_energy = kinetic_energy + potential_energy

    # Normalize velocities for consistent arrow scaling
    velocities_normalized = velocities / np.linalg.norm(velocities, axis=2, keepdims=True)
    velocities_normalized[np.isnan(velocities_normalized)] = 0  # Handle zero velocities

    # Create figure and subplots
    fig, (scatter_ax, energy_ax) = plt.subplots(1, 2, figsize=(14, 6))

    # Configure scatter plot (left subplot)
    scatter_ax.set_aspect('equal')
    x_min, x_max = np.min(Z[:, 0]), np.max(Z[:, 0])
    y_min, y_max = np.min(Z[:, 1]), np.max(Z[:, 1])
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    scatter_ax.set_xlim(x_min - margin, x_max + margin)
    scatter_ax.set_ylim(y_min - margin, y_max + margin)
    scatter_ax.set_title('Agent Movements', fontsize=12)
    scatter_ax.set_xlabel('X')
    scatter_ax.set_ylabel('Y')

    # Particle properties
    dotsize = 600 / N
    dot_colors = 'blue'  # All particles in blue

    scatter = scatter_ax.scatter(
        positions[0, :, 0], positions[0, :, 1],
        s=dotsize, c=dot_colors, alpha=0.6
    )
    quiver = scatter_ax.quiver(
        positions[0, :, 0], positions[0, :, 1],
        velocities_normalized[0, :, 0], velocities_normalized[0, :, 1],
        angles='xy', width=0.001*(30/N), scale=70*(N/30), color='black'  # Arrows in black
    )

    # Configure energy plot (right subplot)
    energy_ax.plot(time, total_energy, label='Total energy', color='green')
    energy_ax.plot(time, kinetic_energy, label='Kinetic energy', color='blue')
    energy_ax.plot(time, potential_energy, label='Potential energy', color='orange')
    energy_ax.set_title('Energy evolution', fontsize=14)
    energy_ax.set_xlabel('Time (s)')
    energy_ax.set_ylabel('Energy')
    energy_ax.legend(fontsize=10, loc='best')  # Legend placed inside the subplot
    energy_ax.grid(True)
    moving_point, = energy_ax.plot([], [], 'ro')  # Moving red point

    # Animation update function
    def update(frame):
        # Update scatter positions
        scatter.set_offsets(positions[frame])

        # Update quiver arrows
        quiver.set_offsets(positions[frame])
        quiver.set_UVC(velocities_normalized[frame, :, 0], velocities_normalized[frame, :, 1])

        # Update moving point on energy plot
        moving_point.set_data(time[frame], total_energy[frame])

        # Update title with current time
        scatter_ax.set_title(f'Agents movements at time: {time[frame]:.3f}s', fontsize=12)
        return scatter, quiver, moving_point

    # Create animation
    ani = FuncAnimation(fig, update, frames=T, interval=200, blit=False)

    plt.tight_layout()

    # Save the animation if requested
    if SAVEFIG:
        mp4_filename = f"{filename_template}.mp4"

        # Save as MP4 using ffmpeg
        print(f"Saving animation as {mp4_filename}...")
        ani.save(mp4_filename, dpi=300, fps=30, writer='ffmpeg')
        print(f"Animation saved as {mp4_filename}")
        
    # Always show the animation
    plt.show()

def make_plot_animation_agents_velocity(positions, velocities, h, Z, SAVEFIG=False, filename_template="animation_agents_velocity"):
    """
    Create an animated plot showing agent movements alongside the evolution of maximum velocity difference.

    Parameters:
    - positions: Array of shape (T, N, 2) containing positions of N agents over T timesteps.
    - velocities: Array of shape (T, N, 2) containing velocities of N agents over T timesteps.
    - h: Time step size.
    - Z: Array of shape (N, 2) containing fixed positions to account for in potential calculations.
    - SAVEFIG: Boolean. If True, saves the animation to a file.
    - filename_template: Template for saving filenames if SAVEFIG=True.

    Returns:
    - An interactive matplotlib animation.
    """
    T, N, _ = positions.shape
    time = np.arange(T) * h

    # Compute maximum velocity differences
    velocity_magnitudes = np.linalg.norm(velocities, axis=2)
    max_velocity_differences = np.max(velocity_magnitudes, axis=1) - np.min(velocity_magnitudes, axis=1)

    # Normalize velocities for consistent arrow scaling
    velocities_normalized = velocities / np.linalg.norm(velocities, axis=2, keepdims=True)
    velocities_normalized[np.isnan(velocities_normalized)] = 0  # Handle zero velocities

    # Create figure and subplots
    fig, (scatter_ax, velocity_ax) = plt.subplots(1, 2, figsize=(14, 6))

    # Configure scatter plot (left subplot)
    scatter_ax.set_aspect('equal')
    x_min, x_max = np.min(Z[:, 0]), np.max(Z[:, 0])
    y_min, y_max = np.min(Z[:, 1]), np.max(Z[:, 1])
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    scatter_ax.set_xlim(x_min - margin, x_max + margin)
    scatter_ax.set_ylim(y_min - margin, y_max + margin)
    scatter_ax.set_title('Agent Movements', fontsize=12)
    scatter_ax.set_xlabel('X')
    scatter_ax.set_ylabel('Y')

    # Particle properties
    dotsize = 600 / N
    dot_colors = 'blue'  # All particles in blue

    scatter = scatter_ax.scatter(
        positions[0, :, 0], positions[0, :, 1],
        s=dotsize, c=dot_colors, alpha=0.6
    )
    quiver = scatter_ax.quiver(
        positions[0, :, 0], positions[0, :, 1],
        velocities_normalized[0, :, 0], velocities_normalized[0, :, 1],
        angles='xy', width=0.001*(30/N), scale=70*(N/30), color='black'  # Arrows in black
    )

    # Configure velocity difference plot (right subplot)
    velocity_ax.plot(time, max_velocity_differences, label='Maximal velocity difference', color='gray')
    velocity_ax.set_title('Maximal velocity difference evolution', fontsize=12)
    velocity_ax.set_xlabel('Time (s)')
    velocity_ax.set_ylabel('Velocity difference')
    velocity_ax.legend(loc='best')  # Legend placed inside the subplot
    velocity_ax.grid(True)
    moving_point, = velocity_ax.plot([], [], 'ro')  # Moving red point

    # Animation update function
    def update(frame):
        # Update scatter positions
        scatter.set_offsets(positions[frame])

        # Update quiver arrows
        quiver.set_offsets(positions[frame])
        quiver.set_UVC(velocities_normalized[frame, :, 0], velocities_normalized[frame, :, 1])

        # Update moving point on velocity difference plot
        moving_point.set_data(time[frame], max_velocity_differences[frame])

        # Update title with current time
        scatter_ax.set_title(f'Agents movements at time: {time[frame]:.3f}s', fontsize=12)
        return scatter, quiver, moving_point

    # Create animation
    ani = FuncAnimation(fig, update, frames=T, interval=200, blit=False)

    plt.tight_layout()

    # Save the animation if requested
    if SAVEFIG:
        mp4_filename = f"{filename_template}.mp4"

        # Save as MP4 using ffmpeg
        print(f"Saving animation as {mp4_filename}...")
        ani.save(mp4_filename, dpi=300, fps=30, writer='ffmpeg')
        print(f"Animation saved as {mp4_filename}")

    # Always show the animation
    plt.show()

def make_plot_animation_agents_min_distance(positions, velocities, h, Z, SAVEFIG=False, filename_template="animation_agents_distance"):
    """
    Create an animated plot showing agent movements with velocity directions and the evolution of minimum distance.

    Parameters:
    - positions: Array of shape (T, N, 2) containing positions of N agents over T timesteps.
    - velocities: Array of shape (T, N, 2) containing velocities of N agents over T timesteps.
    - h: Time step size.
    - Z: Array of shape (N, 2) containing fixed positions to account for in potential calculations.
    - SAVEFIG: Boolean. If True, saves the animation to a file.
    - filename_template: Template for saving filenames if SAVEFIG=True.

    Returns:
    - An interactive matplotlib animation.
    """
    T, N, _ = positions.shape
    time = np.arange(T) * h

    # Compute minimum distances
    min_distances = compute_minimum_distance(positions)

    # Normalize velocities for consistent arrow scaling
    velocities_normalized = velocities / np.linalg.norm(velocities, axis=2, keepdims=True)
    velocities_normalized[np.isnan(velocities_normalized)] = 0  # Handle zero velocities

    # Create figure and subplots
    fig, (scatter_ax, distance_ax) = plt.subplots(1, 2, figsize=(14, 6))

    # Configure scatter plot (left subplot)
    scatter_ax.set_aspect('equal')
    x_min, x_max = np.min(Z[:, 0]), np.max(Z[:, 0])
    y_min, y_max = np.min(Z[:, 1]), np.max(Z[:, 1])
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    scatter_ax.set_xlim(x_min - margin, x_max + margin)
    scatter_ax.set_ylim(y_min - margin, y_max + margin)
    scatter_ax.set_title('Agent Movements', fontsize=12)
    scatter_ax.set_xlabel('X')
    scatter_ax.set_ylabel('Y')

    # Particle properties
    dotsize = 600 / N
    dot_colors = 'blue'  # All particles in blue

    scatter = scatter_ax.scatter(
        positions[0, :, 0], positions[0, :, 1],
        s=dotsize, c=dot_colors, alpha=0.6
    )
    quiver = scatter_ax.quiver(
        positions[0, :, 0], positions[0, :, 1],
        velocities_normalized[0, :, 0], velocities_normalized[0, :, 1],
        angles='xy', width=0.001*(30/N), scale=70*(N/30), color='black'  # Arrows in black
    )

    # Configure minimum distance plot (right subplot)
    distance_ax.plot(time, min_distances, label='Minimum distance', color='purple')
    distance_ax.set_title('Minimum distance evolution', fontsize=12)
    distance_ax.set_xlabel('Time (s)')
    distance_ax.set_ylabel('Distance')
    distance_ax.legend(loc='best')
    distance_ax.grid(True)
    moving_point, = distance_ax.plot([], [], 'ro')  # Moving red point

    # Animation update function
    def update(frame):
        # Update scatter positions
        scatter.set_offsets(positions[frame])

        # Update quiver arrows
        quiver.set_offsets(positions[frame])
        quiver.set_UVC(velocities_normalized[frame, :, 0], velocities_normalized[frame, :, 1])

        # Update moving point on minimum distance plot
        moving_point.set_data(time[frame], min_distances[frame])

        # Update title with current time
        scatter_ax.set_title(f'Agents movements at time: {time[frame]:.3f}s', fontsize=12)
        return scatter, quiver, moving_point

    # Create animation
    ani = FuncAnimation(fig, update, frames=T, interval=200, blit=False)

    plt.tight_layout()

    # Save the animation if requested
    if SAVEFIG:
        mp4_filename = f"{filename_template}.mp4"

        # Save as MP4 using ffmpeg
        print(f"Saving animation as {mp4_filename}...")
        ani.save(mp4_filename, dpi=300, fps=30, writer='ffmpeg')
        print(f"Animation saved as {mp4_filename}")

    # Always show the animation
    plt.show()