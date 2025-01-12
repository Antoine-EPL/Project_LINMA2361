import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

from visualization import * # Import the visualization functions

def adjacency_matrix(graph_type, N):
    """
    Generate an adjacency matrix for a specific graph type.

    Parameters:
    - graph_type: Type of graph to generate the adjacency matrix for.
    - N: Number of agents.

    Returns:
    - A: Adjacency matrix for the specified graph type.
    """
    A = np.zeros((N, N))
    
    if graph_type == "0":
        A[:, :] = 1
        for i in range(N):
            A[i, i] = 0

    elif graph_type == "1":
        for i in range(N-1):
            A[i, i+1] = 1
        A[0, 1:N] = 1
        A[0:int(N/2)-1, int(N/2)-1] = 1
        A[int(N/2)-1, int(N/2):N] = 1
        A[:N-1, N-1] = 1
        A = A + A.T

    elif graph_type == "2":
        for i in range(N-1):
            A[i, i+1] = 1
        A = A + A.T

    elif graph_type == "3":
        for i in range(N-1):
            A[i, i+1] = 1
        A = A + A.T
        for i in range(int(N / 10)):
            A[i * 10, :] = 1
            A[:, i * 10] = 1
            A[i * 10, i * 10] = 0

    return A

def psi_function(r, alpha):
    """
    Singular weight function psi(r) = 1 / r^alpha.

    Parameters:
    - r: Array of distances between agents.
    - alpha: Exponent for the singularity of the weight function.

    Returns:
    - Singular weight values for each distance in r.
    """
    return np.nan_to_num(np.power(r, -alpha))

def phi_function(r, beta):
    """
    Regular weight function phi(r) = 1 / (1 + r)^beta.

    Parameters:
    - r: Array of distances between agents.
    - beta: Exponent controlling the weight's decay with distance.

    Returns:
    - Regular weight values for each distance in r.
    """
    return np.nan_to_num(np.power(1 + r, -beta)) 

def cucker_smale_model_pattern_formation(X, V, Z,adjacency_psi,adjacency_phi, alpha, beta, K, M, N):
    """
    Cucker-Smale model for pattern formation based on bird-like flocking behavior.

    Parameters:
    - X: Positions of the agents (N x 2 array).
    - V: Velocities of the agents (N x 2 array).
    - Z: Desired relative spatial configuration between agents (N x 2 array).
    - adjacency_psi: Adjacency matrix for the singular weight psi (N x N array).
    - adjacency_phi: Adjacency matrix for the regular weight phi (N x N array).
    - alpha: Exponent for the singular weight psi.
    - beta: Exponent for the regular weight phi.
    - K: Coefficient for velocity alignment.
    - M: Coefficient for formation control.
    - N: Number of agents.

    Returns:
    - velocity_updates: Updated velocity changes for all agents (N x 2 array).
    """
    velocity_updates = np.zeros((N, 2)) # Here, d=2 for 2D space

    for i in range(N):
        # Recovery of the set of neighbors for psi
        J_psi = adjacency_psi[i, :] == 1

        # Compute the distance between the agent and its neighbors
        dist_psi=np.sqrt((np.power(X[i][0]-X[:,0],2)+np.power(X[i][1]-X[:,1],2)))[J_psi]

        # To ensure that calculations do not result in undefined or infinite values
        dist_psi=np.nan_to_num(dist_psi)

        # Compute the weight for the singular weight function psi
        weight_psi = psi_function(dist_psi, alpha)
        
        # Velocity alignment term
        velocity_alignment = np.sum((V[J_psi]-V[i])*np.array([weight_psi]).T,axis=0)

        # Normalize the velocity alignment term
        velocity_alignment *= K / N

        # Formation control term
        formation_control = np.zeros(2)

        # Recovery of the set of neighbors for phi
        J_phi = adjacency_phi[i, :] == 1

        # Compute the distance needed for the regular weight function phi
        dist_phi = np.sqrt((np.power(X[i][0]-Z[i][0]-X[:,0]+Z[:,0],2)+np.power(X[i][1]-Z[i][1]-X[:,1]+Z[:,1],2)))[J_phi]

        # To ensure that calculations do not result in undefined or infinite values
        dist_phi = np.nan_to_num(dist_phi)

        # Compute the weight for the regular weight function phi
        weight_phi = phi_function(dist_phi, beta)

        # Compute the formation control term
        formation_control = np.sum((X[J_phi]-Z[J_phi]-X[i]+Z[i])*np.array([weight_phi]).T,axis=0)

        # Normalize the formation control term
        formation_control *= M

        # Combine forces
        total_force = velocity_alignment + formation_control

        # Update the velocity of the agent
        velocity_updates[i] = total_force

    return np.nan_to_num(velocity_updates)

def pattern_formation(curvename,N): 
    """
    Generate curve points for a given curve name.

    Parameters:
    - curvename: Name of the curve to generate. Options are 'circle', 'pi', and 'USA'.
    - N: Number of points to generate on the curve.

    Returns:
    - X: Array of x-coordinates of the curve points.
    - Y: Array of y-coordinates of the curve points.
    """

    dom = 2 * np.pi # All curves are closed, so the range is always 0 to 2*pi. 

    t = np.linspace(0,dom,N,endpoint=False) 

    if curvename=='circle':
        X = np.cos(t) 
        Y = np.sin(t)
    elif curvename=='pi':
        X = 17/31 *np.sin(235/57 - 32 *t) + 19/17 *np.sin(192/55 - 30 *t) + 47/32 *np.sin(69/25 - 29 *t) + 35/26 *np.sin(75/34 - 27 *t) + 6/31 *np.sin(23/10 - 26 *t) + 35/43 *np.sin(10/33 - 25 *t) + 126/43 *np.sin(421/158 - 24 *t) + 143/57 *np.sin(35/22 - 22 *t) + 106/27 *np.sin(84/29 - 21 *t) + 88/25 *np.sin(23/27 - 20 *t) + 74/27 *np.sin(53/22 - 19 *t) + 44/53 *np.sin(117/25 - 18 *t) + 126/25 *np.sin(88/49 - 17 *t) + 79/11 *np.sin(43/26 - 16 *t) + 43/12 *np.sin(41/17 - 15 *t) + 47/27 *np.sin(244/81 - 14 *t) + 8/5 *np.sin(79/19 - 13 *t) + 373/46 *np.sin(109/38 - 12 *t) + 1200/31 *np.sin(133/74 - 11 *t) + 67/24 *np.sin(157/61 - 10 *t) + 583/28 *np.sin(13/8 - 8 *t) + 772/35 *np.sin(59/16 - 7 *t) + 3705/46 *np.sin(117/50 - 6 *t) + 862/13 *np.sin(19/8 - 5 *t) + 6555/34 *np.sin(157/78 - 3 *t) + 6949/13 *np.sin(83/27 - t) - 6805/54 *np.sin(2 *t + 1/145) - 5207/37 *np.sin(4 *t + 49/74) - 1811/58 *np.sin(9 *t + 55/43) - 63/20 *np.sin(23 *t + 2/23) - 266/177 *np.sin(28 *t + 13/18) - 2/21 *np.sin(31 *t + 7/16)
        Y = 70/37 *np.sin(65/32 - 32 *t) + 11/12 *np.sin(98/41 - 31 *t) + 26/29 *np.sin(35/12 - 30 *t) + 54/41 *np.sin(18/7 - 29 *t) + 177/71 *np.sin(51/19 - 27 *t) + 59/34 *np.sin(125/33 - 26 *t) + 49/29 *np.sin(18/11 - 25 *t) + 151/75 *np.sin(59/22 - 24 *t) + 52/9 *np.sin(118/45 - 22 *t) + 52/33 *np.sin(133/52 - 21 *t) + 37/45 *np.sin(61/14 - 20 *t) + 143/46 *np.sin(144/41 - 19 *t) + 254/47 *np.sin(19/52 - 18 *t) + 246/35 *np.sin(92/25 - 17 *t) + 722/111 *np.sin(176/67 - 16 *t) + 136/23 *np.sin(3/19 - 15 *t) + 273/25 *np.sin(32/21 - 13 *t) + 229/33 *np.sin(117/28 - 12 *t) + 19/4 *np.sin(43/11 - 11 *t) + 135/8 *np.sin(23/10 - 10 *t) + 205/6 *np.sin(33/23 - 8 *t) + 679/45 *np.sin(55/12 - 7 *t) + 101/8 *np.sin(11/12 - 6 *t) + 2760/59 *np.sin(40/11 - 5 *t) + 1207/18 *np.sin(21/23 - 4 *t) + 8566/27 *np.sin(39/28 - 3 *t) + 12334/29 *np.sin(47/37 - 2 *t) + 15410/39 *np.sin(185/41 - t) - 596/17 *np.sin(9 *t + 3/26) - 247/28 *np.sin(14 *t + 25/21) - 458/131 *np.sin(23 *t + 21/37) - 41/36 *np.sin(28 *t + 7/8)
    elif curvename=='USA':
        X = -1/7 * np.sin(7/12 - 124 * t) - 3/7 * np.sin(22/15 - 123 * t) - 2/7 * np.sin(9/7 - 119 * t) - 5/9 * np.sin(7/9 - 118 * t) - 3/10 * np.sin(7/9 - 117 * t) - 5/7 * np.sin(4/3 - 111 * t) - 5/8 * np.sin(4/3 - 104 * t) - 5/11 * np.sin(16/13 - 95 * t) - 6/19 * np.sin(10/9 - 92 * t) - 4/5 * np.sin(4/11 - 88 * t) - 18/11 * np.sin(17/11 - 72 * t) - 10/7 * np.sin(1/5 - 71 * t) - 13/10 * np.sin(1/4 - 70 * t) - 7/11 * np.sin(4/3 - 69 * t) - np.sin(1/21 - 68 * t) - 10/13 * np.sin(11/8 - 67 * t) - 23/12 * np.sin(1/3 - 63 * t) - 6/5 * np.sin(3/2 - 62 * t) - 9/7 * np.sin(7/8 - 61 * t) - 7/4 * np.sin(1 - 60 * t) - 15/7 * np.sin(5/7 - 58 * t) - 1/5 * np.sin(8/7 - 56 * t) - 43/17 * np.sin(14/9 - 50 * t) - 25/8 * np.sin(16/17 - 45 * t) - 41/21 * np.sin(4/9 - 44 * t) - 48/19 * np.sin(1/5 - 42 * t) - 15/7 * np.sin(1/18 - 41 * t) - 36/11 * np.sin(1/7 - 36 * t) - 43/16 * np.sin(13/11 - 32 * t) - 12/5 * np.sin(1/6 - 31 * t) - 46/11 * np.sin(3/11 - 27 * t) - 39/8 * np.sin(2/11 - 24 * t) - 56/11 * np.sin(2/9 - 23 * t) - 37/7 * np.sin(14/9 - 16 * t) - 1168/5 * np.sin(5/6 - 2 * t) + 28/5 * np.sin(26 * t) + 41961/40 * np.sin(t + 41/20) + 443/3 * np.sin(3 * t + 29/10) + 1311/13 * np.sin(4 * t + 17/10) + 268/7 * np.sin(5 * t + 22/9) + 86/9 * np.sin(6 * t + 4/7) + 97/6 * np.sin(7 * t + 31/16) + 42/5 * np.sin(8 * t + 41/21) + 240/11 * np.sin(9 * t + 24/7) + 45/4 * np.sin(10 * t + 13/3) + 108/11 * np.sin(11 * t + 5/9) + 531/28 * np.sin(12 * t + 11/8) + 76/11 * np.sin(13 * t + 4/5) + 19/2 * np.sin(14 * t + 60/13) + 235/18 * np.sin(15 * t + 29/8) + 89/8 * np.sin(17 * t + 13/11) + 39/7 * np.sin(18 * t + 29/15) + 122/13 * np.sin(19 * t + 35/8) + 128/15 * np.sin(20 * t + 51/13) + 66/13 * np.sin(21 * t + 29/9) + 52/15 * np.sin(22 * t + 40/11) + 14/3 * np.sin(25 * t + 1/21) + 3/2 * np.sin(28 * t + 1/34) + 19/14 * np.sin(29 * t + 37/8) + 53/9 * np.sin(30 * t + 5/12) + 54/11 * np.sin(33 * t + 18/7) + 37/10 * np.sin(34 * t + 5/2) + 27/7 * np.sin(35 * t + 1/4) + 19/3 * np.sin(37 * t + 9/2) + 16/5 * np.sin(38 * t + 29/8) + 17/13 * np.sin(39 * t + 9/8) + 20/13 * np.sin(40 * t + 5/3) + 26/11 * np.sin(43 * t + 17/18) + 7/4 * np.sin(46 * t + 24/7) + 15/8 * np.sin(47 * t + 9/2) + 8/7 * np.sin(48 * t + 26/9) + 31/15 * np.sin(49 * t + 40/11) + 4/3 * np.sin(51 * t + 13/6) + 4/3 * np.sin(52 * t + 29/11) + 1/2 * np.sin(53 * t + 12/11) + 5/4 * np.sin(54 * t + 30/13) + 9/8 * np.sin(55 * t + 23/12) + 17/18 * np.sin(57 * t + 5/3) + 15/11 * np.sin(59 * t + 94/21) + np.sin(64 * t + 6/7) + 9/5 * np.sin(65 * t + 20/11) + 18/13 * np.sin(66 * t + 1/2) + 7/9 * np.sin(73 * t + 5/2) + 7/6 * np.sin(74 * t + 31/11) + 4/7 * np.sin(75 * t + 10/21) + 5/7 * np.sin(76 * t + 4/7) + 6/13 * np.sin(77 * t + 27/11) + 7/8 * np.sin(78 * t + 31/9) + 3/5 * np.sin(79 * t + 13/6) + 4/5 * np.sin(80 * t + 5/3) + 2/3 * np.sin(81 * t + 16/5) + 6/5 * np.sin(82 * t + 5/7) + 1/5 * np.sin(83 * t + 1/13) + 4/5 * np.sin(84 * t + 21/13) + 4/5 * np.sin(85 * t + 35/12) + 4/7 * np.sin(86 * t + 31/10) + 1/7 * np.sin(87 * t + 47/10) + 1/7 * np.sin(89 * t + 23/5) + 7/10 * np.sin(90 * t + 4/11) + 1/8 * np.sin(91 * t + 25/7) + 3/7 * np.sin(93 * t + 59/13) + 3/5 * np.sin(94 * t + 27/8) + 1/4 * np.sin(96 * t + 8/7) + 3/5 * np.sin(97 * t + 9/7) + 17/18 * np.sin(98 * t + 10/3) + 3/7 * np.sin(99 * t + 1/4) + 1/22 * np.sin(100 * t + 22/5) + 2/3 * np.sin(101 * t + 1) + 1/3 * np.sin(102 * t + 4/7) + 7/11 * np.sin(103 * t + 11/4) + 1/11 * np.sin(105 * t + 53/16) + 1/9 * np.sin(106 * t + 61/13) + 5/9 * np.sin(107 * t + 113/28) + 4/11 * np.sin(108 * t + 5/14) + 1/2 * np.sin(109 * t + 25/9) + 5/14 * np.sin(110 * t + 28/9) + 3/10 * np.sin(112 * t + 8/11) + 1/9 * np.sin(113 * t + 13/10) + 2/5 * np.sin(114 * t + 15/7) + 11/16 * np.sin(115 * t + 19/7) + 2/7 * np.sin(116 * t + 32/7) + 1/4 * np.sin(120 * t + 12/7) + 3/4 * np.sin(121 * t + 11/9) + 10/19 * np.sin(122 * t + 11/9)
        Y = -4/7 * np.sin(4/7 - 123 * t) - 1/6 * np.sin(8/7 - 121 * t) - 2/5 * np.sin(12/11 - 117 * t) - 1/5 * np.sin(4/7 - 116 * t) - 3/13 * np.sin(23/24 - 112 * t) - 6/5 * np.sin(5/8 - 104 * t) - 10/13 * np.sin(8/7 - 103 * t) - 39/40 * np.sin(7/6 - 96 * t) - 4/5 * np.sin(4/13 - 95 * t) - 3/4 * np.sin(7/8 - 92 * t) - 2/9 * np.sin(9/8 - 91 * t) - 8/7 * np.sin(26/17 - 79 * t) - 11/7 * np.sin(1/17 - 76 * t) - 5/6 * np.sin(7/8 - 72 * t) - 5/8 * np.sin(7/5 - 70 * t) - 5/11 * np.sin(14/9 - 68 * t) - 4/7 * np.sin(4/7 - 65 * t) - 7/12 * np.sin(8/7 - 64 * t) - 11/7 * np.sin(9/14 - 62 * t) - 13/5 * np.sin(1 - 57 * t) - 3/5 * np.sin(2/3 - 55 * t) - 1/17 * np.sin(3/2 - 53 * t) - 8/3 * np.sin(1/2 - 46 * t) - 11/12 * np.sin(19/13 - 41 * t) - 5/4 * np.sin(28/19 - 23 * t) + 39/20 * np.sin(43 * t) + 2/9 * np.sin(108 * t) + 4415/7 * np.sin(t + 4/9) + 4997/21 * np.sin(2 * t + 3) + 551/4 * np.sin(3 * t + 11/6) + 455/8 * np.sin(4 * t + 23/6) + 247/10 * np.sin(5 * t + 19/11) + 669/10 * np.sin(6 * t + 23/24) + 377/6 * np.sin(7 * t + 14/5) + 115/8 * np.sin(8 * t + 28/9) + 43/8 * np.sin(9 * t + 31/12) + 219/14 * np.sin(10 * t + 57/28) + 1057/33 * np.sin(11 * t + 19/11) + 43/9 * np.sin(12 * t + 9/14) + 183/7 * np.sin(13 * t + 1/8) + 71/5 * np.sin(14 * t + 6/7) + 211/15 * np.sin(15 * t + 31/8) + 23/3 * np.sin(16 * t + 35/9) + 96/7 * np.sin(17 * t + 25/7) + 13 * np.sin(18 * t + 75/19) + 93/11 * np.sin(19 * t + 9/4) + 90/13 * np.sin(20 * t + 22/7) + 99/8 * np.sin(21 * t + 7/11) + 68/9 * np.sin(22 * t + 12/7) + 74/11 * np.sin(24 * t + 59/20) + 39/7 * np.sin(25 * t + 33/8) + 49/9 * np.sin(26 * t + 16/7) + 19/8 * np.sin(27 * t + 67/17) + 62/13 * np.sin(28 * t + 5/4) + 26/7 * np.sin(29 * t + 16/7) + 19/12 * np.sin(30 * t + 7/9) + 17/8 * np.sin(31 * t + 17/5) + 23/6 * np.sin(32 * t + 39/10) + 73/8 * np.sin(33 * t + 4) + 32/9 * np.sin(34 * t + 19/5) + 3/5 * np.sin(35 * t + 37/19) + 7/9 * np.sin(36 * t + 25/8) + 1/8 * np.sin(37 * t + 67/17) + 17/6 * np.sin(38 * t + 18/5) + 10/3 * np.sin(39 * t + 19/9) + 40/9 * np.sin(40 * t + 19/8) + 3/2 * np.sin(42 * t + 9/4) + 16/7 * np.sin(44 * t + 5/12) + 26/11 * np.sin(45 * t + 3/8) + 15/8 * np.sin(47 * t + 31/12) + 15/8 * np.sin(48 * t + 27/8) + 19/14 * np.sin(49 * t + 15/7) + 13/5 * np.sin(50 * t + 26/9) + 5/3 * np.sin(51 * t + 1/2) + 17/16 * np.sin(52 * t + 9/2) + 9/7 * np.sin(54 * t + 32/7) + 11/8 * np.sin(56 * t + 1/13) + 16/9 * np.sin(58 * t + 101/25) + 68/67 * np.sin(59 * t + 17/4) + 12/13 * np.sin(60 * t + 18/7) + 9/5 * np.sin(61 * t + 20/7) + 8/9 * np.sin(63 * t + 37/8) + 4/3 * np.sin(66 * t + 19/11) + 11/7 * np.sin(67 * t + 14/9) + 8/15 * np.sin(69 * t + 25/9) + 11/9 * np.sin(71 * t + 61/13) + 1/7 * np.sin(73 * t + 59/15) + 5/7 * np.sin(74 * t + 23/13) + 3/2 * np.sin(75 * t + 9/7) + 1/2 * np.sin(77 * t + 7/10) + 2/9 * np.sin(78 * t + 25/6) + 6/7 * np.sin(80 * t + 2/9) + 2/3 * np.sin(81 * t + 25/12) + 1/4 * np.sin(82 * t + 25/7) + 8/7 * np.sin(83 * t + 17/18) + 7/10 * np.sin(84 * t + 21/8) + 11/9 * np.sin(85 * t + 39/10) + 4/7 * np.sin(86 * t + 38/13) + 3/8 * np.sin(87 * t + 42/13) + 3/8 * np.sin(88 * t + 18/7) + 11/21 * np.sin(89 * t + 1/4) + 24/25 * np.sin(90 * t + 13/11) + 2/5 * np.sin(93 * t + 65/16) + 3/7 * np.sin(94 * t + 151/38) + 8/11 * np.sin(97 * t + 1/23) + 20/19 * np.sin(98 * t + 31/16) + 3/4 * np.sin(99 * t + 21/10) + 1/2 * np.sin(100 * t + 9/4) + 7/12 * np.sin(101 * t + 26/11) + 11/16 * np.sin(102 * t + 50/17) + 4/7 * np.sin(105 * t + 43/10) + 1/6 * np.sin(106 * t + 17/6) + 3/10 * np.sin(107 * t + 19/8) + 1/3 * np.sin(109 * t + 51/13) + 3/13 * np.sin(110 * t + 4) + 5/9 * np.sin(111 * t + 50/17) + 3/7 * np.sin(113 * t + 21/8) + 7/15 * np.sin(114 * t + 15/7) + 2/3 * np.sin(115 * t + 5/6) + 7/9 * np.sin(118 * t + 47/12) + 5/6 * np.sin(119 * t + 7/2) + 1/7 * np.sin(120 * t + 30/7) + 1/14 * np.sin(122 * t + 1/3) + 1/3 * np.sin(124 * t + 38/9)
    elif curvename=='moose':
        X = 1/2 * np.sin(55/13 - 100 * t) + 2/7 * np.sin(20/7 - 97 * t) + 8/11 * np.sin(69/17 - 96 * t) + 2/11 * np.sin(4/3 - 93 * t) + 5/11 * np.sin(67/19 - 92 * t) + 4/15 * np.sin(7/11 - 91 * t) + 6/17 * np.sin(5/7 - 88 * t) + 3/5 * np.sin(27/10 - 87 * t) + 1/2 * np.sin(20/11 - 85 * t) + 2/5 * np.sin(32/21 - 83 * t) + 1/5 * np.sin(29/14 - 81 * t) + 4/13 * np.sin(2/5 - 79 * t) + 5/8 * np.sin(5/7 - 77 * t) + 5/7 * np.sin(53/21 - 75 * t) + 2/5 * np.sin(11/7 - 74 * t) + 8/9 * np.sin(18/7 - 73 * t) + 2/5 * np.sin(1/2 - 72 * t) + 11/12 * np.sin(53/18 - 71 * t) + 7/8 * np.sin(10/11 - 69 * t) + 7/5 * np.sin(17/8 - 68 * t) + 11/9 * np.sin(33/8 - 67 * t) + 1/3 * np.sin(25/11 - 66 * t) + 7/10 * np.sin(34/11 - 65 * t) + 7/4 * np.sin(14/9 - 63 * t) + np.sin(31/10 - 62 * t) + np.sin(1/8 - 61 * t) + 4/9 * np.sin(12/7 - 60 * t) +  17/10 * np.sin(46/11 - 59 * t) + 8/9 * np.sin(11/8 - 58 * t) + 7/9 * np.sin(1/9 - 57 * t) + 5/8 * np.sin(37/13 - 56 * t) + 1/20 * np.sin(53/16 - 55 * t) + 27/16 * np.sin(7/6 - 54 * t) + 16/9 * np.sin(17/4 - 53 * t) + 16/11 * np.sin(26/9 - 51 * t) + 28/11 * np.sin(37/8 - 50 * t) + 17/9 * np.sin(14/9 - 49 * t) + 1/8 * np.sin(3/5 - 48 * t) + 7/11 * np.sin(10/3 - 47 * t) + 7/8 * np.sin(33/14 - 46 * t) + 8/5 * np.sin(67/15 - 45 * t) + 3/5 * np.sin(29/10 - 43 * t) + 53/16 * np.sin(37/11 - 42 * t) + 29/9 * np.sin(7/11 - 41 * t) + 59/16 * np.sin(3 - 40 * t) + 14/9 * np.sin(86/19 - 39 * t) + 17/10 * np.sin(11/4 - 38 * t) + 29/8 * np.sin(43/17 - 34 * t) + 21/8 * np.sin(41/11 - 33 * t) + 30/7 * np.sin(5/11 - 32 * t) + 13/9 * np.sin(37/8 - 31 * t) + 52/9 * np.sin(1/3 - 28 * t) + 62/13 * np.sin(21/5 - 27 * t) + 38/13 * np.sin(13/3 - 26 * t) + 11/4 * np.sin(28/11 - 25 * t) + 48/13 * np.sin(1/4 - 23 * t) + 19/11 * np.sin(4/3 - 22 * t) + 49/11 * np.sin(31/9 - 21 * t) + 17/3 * np.sin(1/11 - 20 * t) + 10 * np.sin(17/6 - 18 * t) + 85/6 * np.sin(11/6 - 16 * t) + 228/11 * np.sin(6/13 - 15 * t) + 9/10 * np.sin(12/11 - 14 * t) + 244/13 * np.sin(10/7 - 13 * t) + 40/7 * np.sin(1/4 - 11 * t) + 433/17 * np.sin(1/5 - 10 * t) + 111/5 * np.sin(27/8 - 9 * t) + 871/29 * np.sin(1/22 - 7 * t) + 292/15 * np.sin(19/7 - 6 * t) + 127/11 * np.sin(74/17 - 5 * t) + 1150/19 * np.sin(1/10 - 4 * t) + 211/6 * np.sin(21/10 - 3 * t) + 1077/8 * np.sin(9/5 - 2 * t) + 3027/7 * np.sin(57/14 - t) - 95/3 * np.sin(8 * t + 1/4) - 49/5 * np.sin(12 * t + 7/9) - 107/9 * np.sin(17 * t + 5/6) - 57/14 * np.sin(19 * t + 5/4) - 15/4 * np.sin(24 * t + 7/9) - 65/11 * np.sin(29 * t + 1/31) - 43/9 * np.sin(30 * t + 15/14) - 8/3 * np.sin(35 * t + 37/38) - 137/46 * np.sin(36 * t + 3/2) - 3/7 * np.sin(37 * t + 5/7) - 31/14 * np.sin(44 * t + 5/4) - 5/3 * np.sin(52 * t + 6/13) - 7/5 * np.sin(64 * t + 7/9) - 7/12 * np.sin(70 * t + 1/8) - 1/2 * np.sin(76 * t + 7/8) - 3/8 * np.sin(78 * t + 1/16) - 1/3 * np.sin(80 * t + 19/15) - 9/14 * np.sin(82 * t + 1/9) - 7/9 * np.sin(84 * t + 56/55) - 28/29 * np.sin(86 * t + 5/7) - 1/3 * np.sin(89 * t + 5/12) - 2/7 * np.sin(90 * t + 9/14) - 1/33 * np.sin(94 * t + 3/11) - 1/11 * np.sin(95 * t + 11/9) - 1/15 * np.sin(98 * t + 13/11) - 1/12 * np.sin(99 * t + 1/2)
        Y = 1/5 * np.sin(17/10 - 100 * t) + 3/13 * np.sin(7/12 - 99 * t) + 1/4 * np.sin(37/15 - 98 * t) + 2/5 * np.sin(25/11 - 96 * t) + 5/8 * np.sin(13/7 - 94 * t) + 5/12 * np.sin(41/10 - 93 * t) + 1/2 * np.sin(3/8 - 92 * t) + 5/11 * np.sin(15/4 - 91 * t) + 1/16 * np.sin(21/5 - 90 * t) + 3/8 * np.sin(31/8 - 88 * t) + 4/7 * np.sin(2/11 - 87 * t) + 1/21 * np.sin(19/7 - 86 * t) + 3/8 * np.sin(22/9 - 84 * t) + 1/7 * np.sin(4/9 - 83 * t) + 4/9 * np.sin(29/13 - 82 * t) + 1/20 * np.sin(29/7 - 80 * t) + 11/17 * np.sin(61/15 - 79 * t) + 5/7 * np.sin(26/11 - 78 * t) + 5/9 * np.sin(19/13 - 76 * t) + 4/7 * np.sin(41/10 - 75 * t) + 22/21 * np.sin(9/10 - 74 * t) + 15/16 * np.sin(52/17 - 73 * t) + 5/11 * np.sin(17/18 - 72 * t) + 4/9 * np.sin(10/7 - 71 * t) + 4/13 * np.sin(44/15 - 70 * t) + 2/3 * np.sin(16/9 - 69 * t) + 5/7 * np.sin(56/13 - 68 * t) + 14/11 * np.sin(26/9 - 66 * t) + 7/8 * np.sin(74/21 - 63 * t) + 26/11 * np.sin(29/12 - 61 * t) + 49/48 * np.sin(7/13 - 59 * t) + 13/7 * np.sin(35/9 - 58 * t) + 4/5 * np.sin(28/9 - 57 * t) + 33/14 * np.sin(32/11 - 55 * t) + 9/5 * np.sin(23/24 - 53 * t) + 95/48 * np.sin(19/5 - 52 * t) + 89/22 * np.sin(20/13 - 50 * t) + 93/23 * np.sin(55/12 - 49 * t) + 26/5 * np.sin(15/13 - 48 * t) + 47/12 * np.sin(13/3 - 47 * t) + 23/12 * np.sin(11/16 - 46 * t) + 13/9 * np.sin(31/9 - 45 * t) + 19/9 * np.sin(7/10 - 44 * t) + 15/7 * np.sin(5/3 - 42 * t) + 21/8 * np.sin(19/8 - 40 * t) + 43/11 * np.sin(8/9 - 38 * t) + 26/11 * np.sin(47/11 - 37 * t) + 30/7 * np.sin(8/7 - 36 * t) + 19/8 * np.sin(13/3 - 35 * t) + 11/5 * np.sin(47/13 - 34 * t) + 53/15 * np.sin(11/8 - 33 * t) + 7/4 * np.sin(51/14 - 31 * t) + 53/15 * np.sin(5/12 - 30 * t) + 41/13 * np.sin(26/7 - 29 * t) + 30/7 * np.sin(19/9 - 28 * t) + 37/7 * np.sin(1/10 - 27 * t) + 43/17 * np.sin(1/10 - 26 * t) + 38/37 * np.sin(19/5 - 25 * t) + 26/7 * np.sin(13/11 - 24 * t) + 97/15 * np.sin(27/10 - 23 * t) + 43/5 * np.sin(2 - 22 * t) + 1/10 * np.sin(7/2 - 21 * t) + 17/10 * np.sin(21/8 - 20 * t) + 32/9 * np.sin(19/13 - 16 * t) + 21/4 * np.sin(17/7 - 15 * t) + 51/5 * np.sin(12/5 - 12 * t) + 198/13 * np.sin(7/5 - 11 * t) + 21/4 * np.sin(9/5 - 10 * t) + 669/8 * np.sin(39/10 - 8 * t) + 499/4 * np.sin(11/3 - 7 * t) + 283/9 * np.sin(18/5 - 6 * t) + 255/7 * np.sin(3/10 - 4 * t) + 563/6 * np.sin(26/11 - 3 * t) + 1225/9 * np.sin(47/24 - 2 * t) - 12407/24 * np.sin(t + 11/9) - 112/3 * np.sin(5 * t + 7/12) - 116/5 * np.sin(9 * t + 4/3) - 18/7 * np.sin(13 * t + 25/19) - 121/9 * np.sin(14 * t + 6/5) - 55/12 * np.sin(17 * t + 1/43) - 14/5 * np.sin(18 * t + 5/8) - 81/40 * np.sin(19 * t + 1/23) - 16/5 * np.sin(32 * t + 7/5) - 32/13 * np.sin(39 * t + 11/8) - 40/9 * np.sin(41 * t + 6/5) - 14/5 * np.sin(43 * t + 14/11) - 13/5 * np.sin(51 * t + 5/8) - 20/11 * np.sin(54 * t + 48/47) - 35/36 * np.sin(56 * t + 2/7) - 3/2 * np.sin(60 * t + 13/9) - 18/11 * np.sin(62 * t + 7/10) - 1/4 * np.sin(64 * t + 11/8) - 19/11 * np.sin(65 * t + 7/13) - 3/8 * np.sin(67 * t + 10/19) - 2/7 * np.sin(77 * t + 1/10) - 1/8 * np.sin(81 * t + 10/7) - 4/13 * np.sin(85 * t + 6/7) - 5/14 * np.sin(89 * t + 1/16) - 3/5 * np.sin(95 * t + 7/6) - 1/2 * np.sin(97 * t + 3/7)
    elif curvename=='pig':
        X = (1/4) * np.sin(13/6 - 119 * t) + (1/3) * np.sin(1 - 118 * t) + (2/5) * np.sin(4/3 - 116 * t) + (1/3) * np.sin(9/7 - 115 * t) + (1/6) * np.sin(17/6 - 114 * t) + (1/4) * np.sin(27/14 - 113 * t) + (1/4) * np.sin(23/8 - 112 * t) + (2/5) * np.sin(17/7 - 111 * t) + (1/2) * np.sin(7/2 - 109 * t) + (1/4) * np.sin(2 - 108 * t) + (3/5) * np.sin(3 - 106 * t) + (1/3) * np.sin(13/6 - 105 * t) + (1/14) * np.sin(25/8 - 103 * t) + (2/3) * np.sin(13/3 - 102 * t) + (2/3) * np.sin(17/5 - 101 * t) + (4/5) * np.sin(14/3 - 99 * t) + (1/6) * np.sin(32/7 - 94 * t) + (3/5) * np.sin(4/3 - 93 * t) + (1/2) * np.sin(23/6 - 91 * t) + (12/13) * np.sin(5/6 - 90 * t) + (3/5) * np.sin(33/7 - 89 * t) + (1/3) * np.sin(4 - 88 * t) + (1/3) * np.sin(11/4 - 86 * t) + (4/5) * np.sin(1/4 - 85 * t) + (2/3) * np.sin(7/3 - 83 * t) + (3/5) * np.sin(11/5 - 81 * t) + (9/10) * np.sin(13/7 - 80 * t) + (4/5) * np.sin(14/3 - 79 * t) + (1/2) * np.sin(7/5 - 78 * t) + (1/4) * np.sin(18/7 - 77 * t) + (5/7) * np.sin(25/7 - 76 * t) + (8/7) * np.sin(1/5 - 75 * t) + (1/2) * np.sin(11/4 - 73 * t) + (11/10) * np.sin(1/6 - 72 * t) + (1/9) * np.sin(52/17 - 71 * t) + (3/5) * np.sin(4/3 - 70 * t) + (5/6) * np.sin(14/13 - 69 * t) + (5/6) * np.sin(16/7 - 68 * t) + (1/2) * np.sin(20/7 - 67 * t) + (7/8) * np.sin(5/4 - 66 * t) + (2/5) * np.sin(1/3 - 65 * t) + (3/5) * np.sin(3/4 - 63 * t) + (3/4) * np.sin(7/4 - 62 * t) + (1/2) * np.sin(9/4 - 61 * t) + (4/3) * np.sin(11/5 - 60 * t) + (9/4) * np.sin(12/5 - 59 * t) + (3/4) * np.sin(23/5 - 57 * t) + (3/2) * np.sin(23/6 - 56 * t) + (10/7) * np.sin(65/16 - 55 * t) + (4/7) * np.sin(1/13 - 54 * t) + (3/2) * np.sin(32/7 - 50 * t) + (8/7) * np.sin(9/4 - 49 * t) + (11/7) * np.sin(23/11 - 48 * t) + 2 * np.sin(39/10 - 45 * t) + (5/3) * np.sin(13/3 - 44 * t) + (7/5) * np.sin(2 - 42 * t) + (10/11) * np.sin(17/8 - 41 * t) + (5/2) * np.sin(1/2 - 40 * t) + (6/5) * np.sin(9/5 - 39 * t) + (7/3) * np.sin(19/5 - 38 * t) + (6/7) * np.sin(1/3 - 37 * t) + np.sin(4/3 - 36 * t) + np.sin(12/5 - 35 * t) + (13/5) * np.sin(2/3 - 34 * t) + (19/6) * np.sin(11/4 - 33 * t) + (61/15) * np.sin(14/5 - 32 * t) + (11/12) * np.sin(9/4 - 31 * t) + (37/7) * np.sin(16/5 - 30 * t) + (54/11) * np.sin(17/5 - 29 * t) + 7 * np.sin(17/4 - 26 * t) + (13/7) * np.sin(1/3 - 25 * t) + (61/12) * np.sin(1/17 - 24 * t) + (14/5) * np.sin(47/12 - 23 * t) + (85/12) * np.sin(13/12 - 22 * t) + (45/7) * np.sin(5/6 - 21 * t) + (11/4) * np.sin(17/7 - 18 * t) + (58/7) * np.sin(2 - 17 * t) + (41/2) * np.sin(2/5 - 15 * t) + (61/4) * np.sin(3/2 - 14 * t) + (16/3) * np.sin(13/5 - 13 * t) + (43/7) * np.sin(3 - 11 * t) + (129/5) * np.sin(12/5 - 10 * t) + (209/4) * np.sin(5/7 - 7 * t) + (391/7) * np.sin(23/6 - 5 * t) + (1189/11) * np.sin(14/3 - 3 * t) + (274) * np.sin(3/5 - 2 * t) + (2329/2) * np.sin(10/3 - t) - (78/5) * np.sin(16 * t) - (3/4) * np.sin(82 * t) - (2/5) * np.sin(95 * t) - (1/6) * np.sin(110 * t) - (283/4) * np.sin(4 * t + 10/7) - (105/4) * np.sin(6 * t + 3/2) - (155/4) * np.sin(8 * t + 1/5) - (76/5) * np.sin(9 * t + 3/5) - (40/3) * np.sin(12 * t + 25/24) - (43/4) * np.sin(19 * t + 6/7) - (22/5) * np.sin(20 * t + 5/4) - (13/3) * np.sin(27 * t + 4/5) - (19/5) * np.sin(28 * t + 1/7) - (8/3) * np.sin(43 * t + 1/4) - (13/5) * np.sin(46 * t + 1/2) - (7/8) * np.sin(47 * t + 1) - (2/3) * np.sin(51 * t + 3/5) - (2/5) * np.sin(52 * t + 4/3) - (7/8) * np.sin(53 * t + 12/13) - (1/15) * np.sin(58 * t + 3/2) - np.sin(64 * t + 5/4) - (2/3) * np.sin(74 * t + 10/7) - (3/4) * np.sin(84 * t + 4/5) - (1/2) * np.sin(87 * t + 1/25) - (3/4) * np.sin(92 * t + 10/11) - (1/3) * np.sin(96 * t + 6/5) - (2/3) * np.sin(97 * t + 3/4) - (2/5) * np.sin(100 * t + 4/5) - (1/7) * np.sin(107 * t + 16/15) - (1/4) * np.sin(117 * t + 1/4) - (1/6) * np.sin(120 * t + 4/3)
        Y = (1/11) * np.sin(23/12 - 119 * t) + (1/3) * np.sin(26/9 - 118 * t) + (2/7) * np.sin(7/3 - 117 * t) + (2/7) * np.sin(11/12 - 116 * t) + (1/4) * np.sin(13/3 - 115 * t) + (1/3) * np.sin(11/3 - 114 * t) + (2/5) * np.sin(22/5 - 113 * t) + (1/3) * np.sin(9/2 - 112 * t) + (1/10) * np.sin(1/7 - 110 * t) + (1/4) * np.sin(14/3 - 109 * t) + (1/2) * np.sin(2 - 107 * t) + (1/4) * np.sin(10/11 - 106 * t) + (3/7) * np.sin(1/12 - 102 * t) + (2/5) * np.sin(1/4 - 100 * t) + (2/3) * np.sin(3/5 - 99 * t) + (1/4) * np.sin(13/6 - 98 * t) + (2/5) * np.sin(7/3 - 97 * t) + (1/3) * np.sin(1 - 96 * t) + (1/3) * np.sin(1/3 - 95 * t) + (1/3) * np.sin(37/9 - 94 * t) + (1/2) * np.sin(22/5 - 93 * t) + (2/5) * np.sin(2/5 - 91 * t) + (1/3) * np.sin(4/3 - 90 * t) + (2/3) * np.sin(1 - 89 * t) + (7/8) * np.sin(8/7 - 88 * t) + (1/5) * np.sin(7/3 - 87 * t) + np.sin(29/10 - 86 * t) + (6/5) * np.sin(7/3 - 85 * t) + (1/4) * np.sin(4 - 84 * t) + (2/5) * np.sin(29/7 - 83 * t) + (2/5) * np.sin(11/3 - 82 * t) + (8/7) * np.sin(19/5 - 81 * t) + (1/4) * np.sin(23/5 - 80 * t) + (1/2) * np.sin(15/7 - 79 * t) + (3/5) * np.sin(9/5 - 78 * t) + (2/7) * np.sin(22/5 - 76 * t) + (11/12) * np.sin(20/7 - 75 * t) + (1/2) * np.sin(5/4 - 73 * t) + (3/5) * np.sin(17/4 - 72 * t) + (3/5) * np.sin(9/2 - 71 * t) + (1/4) * np.sin(9/4 - 70 * t) + (3/5) * np.sin(4 - 69 * t) + (3/7) * np.sin(3/5 - 68 * t) + (5/6) * np.sin(7/6 - 67 * t) + (2/3) * np.sin(31/8 - 63 * t) + (1/2) * np.sin(2/5 - 61 * t) + (1/5) * np.sin(23/5 - 60 * t) + (3/4) * np.sin(1/3 - 59 * t) + (4/7) * np.sin(11/3 - 58 * t) + (1/4) * np.sin(10/3 - 57 * t) + (5/3) * np.sin(3/7 - 56 * t) + (1/6) * np.sin(23/12 - 55 * t) + np.sin(33/7 - 54 * t) + (4/3) * np.sin(9/2 - 53 * t) + (11/5) * np.sin(34/11 - 52 * t) + (1/4) * np.sin(19/7 - 51 * t) + (5/3) * np.sin(13/7 - 47 * t) + (7/4) * np.sin(1/2 - 46 * t) + (8/3) * np.sin(1 - 44 * t) + (7/6) * np.sin(7/2 - 43 * t) + (5/2) * np.sin(19/6 - 42 * t) + (13/6) * np.sin(4/3 - 41 * t) + (18/5) * np.sin(12/5 - 40 * t) + 2 * np.sin(13/5 - 39 * t) + (7/3) * np.sin(17/5 - 37 * t) + 3 * np.sin(13/6 - 34 * t) + (15/4) * np.sin(6/5 - 32 * t) + 2 * np.sin(5/3 - 31 * t) + (37/6) * np.sin(14/3 - 30 * t) + (19/4) * np.sin(4/3 - 29 * t) + (25/7) * np.sin(6/5 - 28 * t) + (34/5) * np.sin(13/3 - 25 * t) + (16/5) * np.sin(33/8 - 24 * t) + (4/3) * np.sin(4/5 - 23 * t) + (22/3) * np.sin(97/24 - 22 * t) + (10/9) * np.sin(18/5 - 19 * t) + (210/19) * np.sin(1/14 - 18 * t) + (25/2) * np.sin(4/5 - 16 * t) + (13/4) * np.sin(9/10 - 15 * t) + (79/5) * np.sin(8/3 - 14 * t) + (201/10) * np.sin(11/7 - 13 * t) + 22 * np.sin(17/5 - 12 * t) + (211/4) * np.sin(7/2 - 11 * t) + (112/3) * np.sin(16/5 - 10 * t) + (287/11) * np.sin(9/4 - 9 * t) + (286/5) * np.sin(3 - 6 * t) + (341/6) * np.sin(13/4 - 5 * t) + (97/7) * np.sin(23/5 - 4 * t) + (746/5) * np.sin(13/4 - 3 * t) + (1714/7) * np.sin(7/3 - 2 * t) - (36/5) * np.sin(17 * t) - (10/11) * np.sin(35 * t) - (3079/5) * np.sin(t + 8/7) - (167/3) * np.sin(7 * t + 5/4) - (296/7) * np.sin(8 * t + 1/4) - (29/7) * np.sin(20 * t + 1/6) - 3 * np.sin(21 * t + 5/7) - (13/7) * np.sin(26 * t + 1/5) - (11/5) * np.sin(27 * t + 5/6) - (31/15) * np.sin(33 * t + 4/7) - (39/19) * np.sin(36 * t + 3/5) - (5/2) * np.sin(38 * t + 3/4) - (8/5) * np.sin(45 * t + 3/5) - (7/4) * np.sin(48 * t + 7/6) - (6/5) * np.sin(49 * t + 2/7) - (2/5) * np.sin(50 * t + 5/4) - (4/7) * np.sin(62 * t + 1/5) - (1/5) * np.sin(64 * t + 1/13) - (5/4) * np.sin(65 * t + 3/4) - (2/3) * np.sin(66 * t + 3/2) - (1/14) * np.sin(74 * t + 3/2) - (6/5) * np.sin(77 * t + 1/3) - (2/5) * np.sin(92 * t + 1/4) - (1/5) * np.sin(101 * t + 1/3) - (1/4) * np.sin(103 * t + 5/6) - (1/3) * np.sin(104 * t + 6/5) - (3/5) * np.sin(105 * t + 5/4) - (1/7) * np.sin(108 * t + 4/3) - (1/3) * np.sin(111 * t + 1/4) - (1/4) * np.sin(120 * t + 4/5)
    elif curvename=='camel':
        X = (1/40) * np.sin(23/9 - 100 * t) + (1/6) * np.sin(139/46 - 97 * t) + (1/6) * np.sin(41/10 - 96 * t) + (1/8) * np.sin(13/3 - 95 * t) + (1/6) * np.sin(3 - 93 * t) + (1/6) * np.sin(45/13 - 92 * t) + (1/6) * np.sin(9/2 - 91 * t) + (1/5) * np.sin(9/7 - 90 * t) + (2/7) * np.sin(11/7 - 89 * t) + (1/7) * np.sin(20/9 - 88 * t) + (1/4) * np.sin(11/8 - 87 * t) + (1/4) * np.sin(7/13 - 86 * t) + (1/17) * np.sin(19/5 - 85 * t) + (1/5) * np.sin(17/4 - 84 * t) + (1/4) * np.sin(29/8 - 82 * t) + (1/5) * np.sin(29/10 - 81 * t) + (1/4) * np.sin(4/9 - 79 * t) + (3/7) * np.sin(5/8 - 78 * t) + (1/11) * np.sin(1/5 - 77 * t) + (1/3) * np.sin(63/16 - 74 * t) + (4/9) * np.sin(21/5 - 73 * t) +  (4/9) * np.sin(25/6 - 72 * t) + (1/4) * np.sin(11/6 - 71 * t) + (1/35) * np.sin(46/13 - 70 * t) + (1/4) * np.sin(13/6 - 69 * t) + (1/5) * np.sin(1/2 - 66 * t) + (3/7) * np.sin(9/2 - 65 * t) + (1/3) * np.sin(14/3 - 64 * t) + (6/11) * np.sin(5/2 - 63 * t) + (1/6) * np.sin(1/3 - 62 * t) + (2/7) * np.sin(23/7 - 61 * t) + (5/11) * np.sin(2/7 - 59 * t) + (1/3) * np.sin(33/7 - 58 * t) + (1/6) * np.sin(19/6 - 57 * t) + (5/9) * np.sin(1/7 - 56 * t) + (11/12) * np.sin(25/6 - 53 * t) + (5/6) * np.sin(33/10 - 51 * t) + (4/7) * np.sin(16/7 - 50 * t) + (19/13) * np.sin(13/5 - 48 * t) + (3/7) * np.sin(61/30 - 47 * t) + (4/9) * np.sin(24/7 - 46 * t) + (11/10) * np.sin(1/7 - 45 * t) + (1/3) * np.sin(1/11 - 44 * t) + (12/7) * np.sin(29/7 - 43 * t) + (2/7) * np.sin(35/8 - 42 * t) + (3/8) * np.sin(11/10 - 41 * t) + (3/4) * np.sin(7/3 - 40 * t) + (31/16) * np.sin(8/9 - 39 * t) + (12/7) * np.sin(1/10 - 38 * t) + (4/5) * np.sin(7/9 - 37 * t) + (12/13) * np.sin(10/3 - 36 * t) + (16/9) * np.sin(41/9 - 34 * t) + (7/5) * np.sin(26/9 - 33 * t) + (11/10) * np.sin(13/10 - 29 * t) + (12/5) * np.sin(25/6 - 27 * t) + (50/17) * np.sin(7/6 - 26 * t) + (13/8) * np.sin(55/18 - 25 * t) + (23/8) * np.sin(33/8 - 24 * t) + (31/7) * np.sin(4/3 - 23 * t) + (19/7) * np.sin(37/9 - 22 * t) + (15/4) * np.sin(22/5 - 21 * t) + (37/7) * np.sin(9/2 - 20 * t) + (17/8) * np.sin(14/3 - 19 * t) + (3/4) * np.sin(5/2 - 18 * t) + (43/8) * np.sin(16/5 - 17 * t) + (37/4) * np.sin(3/5 - 16 * t) + (52/9) * np.sin(14/9 - 15 * t) + (11/4) * np.sin(9/5 - 13 * t) + (17/10) * np.sin(13/3 - 11 * t) + (201/7) * np.sin(10/3 - 10 * t) + (19/5) * np.sin(49/16 - 9 * t) + (101/8) * np.sin(24/7 - 8 * t) + (110/13) * np.sin(95/24 - 7 * t) + (112/3) * np.sin(21/5 - 5 * t) + (23/4) * np.sin(19/6 - 4 * t) + (435/11) * np.sin(11/6 - 3 * t) + (3992/13) * np.sin(67/22 - t) - (1314/11) * np.sin(2 * t + 3/4) - (125/9) * np.sin(6 * t + 15/14) - (26/5) * np.sin(12 * t + 1/2) - (141/20) * np.sin(14 * t + 2/3) - (17/4) * np.sin(28 * t + 3/4) - (13/6) * np.sin(30 * t + 2/7) - (10/7) * np.sin(31 * t + 1/2) - (7/6) * np.sin(32 * t + 7/6) - (3/7) * np.sin(35 * t + 15/14) - (2/7) * np.sin(49 * t + 5/8) - (3/5) * np.sin(52 * t + 1/24) - (2/5) * np.sin(54 * t + 4/3) - (3/8) * np.sin(55 * t + 3/2) - (2/5) * np.sin(60 * t + 5/8) - (1/4) * np.sin(67 * t + 1/4) - (3/5) * np.sin(68 * t + 1/7) - (1/11) * np.sin(75 * t + 1) - (2/7) * np.sin(76 * t + 5/9) - (1/4) * np.sin(80 * t + 3/5) - (3/8) * np.sin(83 * t + 7/10) - (1/4) * np.sin(94 * t + 10/11) - (1/23) * np.sin(98 * t + 1/3) - (1/12) * np.sin(99 * t + 3/2) - 685/8
        Y = (1/5) * np.sin(3 - 100 * t) + (1/6) * np.sin(7/3 - 99 * t) + (2/7) * np.sin(2/3 - 98 * t) + (1/6) * np.sin(19/6 - 97 * t) + (1/7) * np.sin(13/4 - 96 * t) + (1/18) * np.sin(31/7 - 95 * t) + (1/14) * np.sin(7/4 - 93 * t) + (1/6) * np.sin(12/5 - 92 * t) + (2/9) * np.sin(13/4 - 91 * t) + (2/7) * np.sin(29/14 - 90 * t) + (1/6) * np.sin(8/3 - 89 * t) + (1/5) * np.sin(23/24 - 88 * t) + (1/7) * np.sin(7/6 - 87 * t) + (2/9) * np.sin(23/7 - 86 * t) + (1/6) * np.sin(7/3 - 85 * t) + (1/7) * np.sin(1/5 - 83 * t) + (1/4) * np.sin(1/2 - 82 * t) + (1/3) * np.sin(10/3 - 81 * t) + (1/5) * np.sin(11/9 - 80 * t) + (3/8) * np.sin(23/9 - 79 * t) + (1/4) * np.sin(9/7 - 78 * t) + (1/3) * np.sin(23/22 - 77 * t) + (1/7) * np.sin(38/11 - 76 * t) + (1/5) * np.sin(11/5 - 75 * t) + (1/6) * np.sin(1/12 - 74 * t) + (1/4) * np.sin(7/5 - 72 * t) + (3/8) * np.sin(35/9 - 71 * t) + (2/7) * np.sin(15/7 - 70 * t) + (6/11) * np.sin(21/10 - 69 * t) + (2/7) * np.sin(5/9 - 68 * t) + (1/2) * np.sin(3/5 - 67 * t) + (1/7) * np.sin(37/10 - 66 * t) + (2/9) * np.sin(6/5 - 65 * t) + (3/7) * np.sin(17/7 - 64 * t) + (6/7) * np.sin(14/3 - 63 * t) + (2/5) * np.sin(13/4 - 62 * t) + (2/7) * np.sin(91/23 - 61 * t) + (2/7) * np.sin(8/5 - 60 * t) + (9/10) * np.sin(13/7 - 59 * t) + (6/13) * np.sin(1/14 - 58 * t) + (1/10) * np.sin(29/8 - 56 * t) + (1/24) * np.sin(23/7 - 55 * t) + (23/22) * np.sin(15/7 - 54 * t) + (3/4) * np.sin(17/5 - 52 * t) + (6/11) * np.sin(19/6 - 51 * t) + (6/11) * np.sin(7/10 - 50 * t) + (2/3) * np.sin(19/8 - 49 * t) + (2/3) * np.sin(11/7 - 48 * t) + (1/7) * np.sin(6/13 - 47 * t) + (2/7) * np.sin(50/11 - 46 * t) + (4/7) * np.sin(8/5 - 45 * t) + (15/14) * np.sin(4/3 - 44 * t) + (19/20) * np.sin(44/15 - 42 * t) + (1/2) * np.sin(12/5 - 41 * t) + (4/7) * np.sin(1 - 40 * t) + (1/7) * np.sin(27/7 - 39 * t) + (13/5) * np.sin(11/7 - 38 * t) + (13/10) * np.sin(16/5 - 36 * t) + (7/5) * np.sin(8/9 - 35 * t) + (8/9) * np.sin(19/20 - 33 * t) + (19/13) * np.sin(25/6 - 32 * t) + (9/7) * np.sin(10/7 - 31 * t) + np.sin(13/5 - 30 * t) + (21/10) * np.sin(11/10 - 28 * t) + (3/2) * np.sin(13/8 - 27 * t) + (3/4) * np.sin(23/8 - 26 * t) + (11/4) * np.sin(17/6 - 25 * t) + (19/7) * np.sin(11/4 - 23 * t) + (11/4) * np.sin(17/6 - 21 * t) + (17/4) * np.sin(16/11 - 20 * t) + (29/11) * np.sin(3/8 - 18 * t) + (1/8) * np.sin(47/12 - 17 * t) + (58/7) * np.sin(11/8 - 16 * t) + (27/2) * np.sin(13/4 - 15 * t) + (53/8) * np.sin(4/5 - 14 * t) + (186/11) * np.sin(22/9 - 12 * t) + (97/13) * np.sin(12/5 - 10 * t) + (191/19) * np.sin(1/34 - 9 * t) + (92/7) * np.sin(21/5 - 8 * t) + 17 * np.sin(24/7 - 7 * t) + (547/7) * np.sin(6/5 - 6 * t) + (805/9) * np.sin(13/6 - 5 * t) + (266/3) * np.sin(18/7 - 4 * t) + (81/5) * np.sin(2/7 - 3 * t) + (1138/7) * np.sin(2/3 - 2 * t) + (956/3) * np.sin(22/5 - t) - (169/12) * np.sin(11 * t) - 2 * np.sin(29 * t) - (39/4) * np.sin(13 * t + 11/8) - (30/7) * np.sin(19 * t + 6/5) - (13/8) * np.sin(22 * t + 7/6) - (49/10) * np.sin(24 * t + 4/9) - (4/3) * np.sin(34 * t + 7/10) - (5/9) * np.sin(37 * t + 4/7) - (1/3) * np.sin(43 * t + 1/2) - (4/7) * np.sin(53 * t + 5/4) - (1/11) * np.sin(57 * t + 1/9) - (3/4) * np.sin(73 * t + 13/9) - (1/5) * np.sin(84 * t + 2/3) - (1/10) * np.sin(94 * t + 5/7) - 28/3
    elif curvename=='omega':
        X = (6/29) * np.sin(73/26 - 78 * t) + (9/41) * np.sin(7/16 - 77 * t) + (3/22) * np.sin(11/3 - 74 * t) + (9/44) * np.sin(53/107 - 73 * t) + (7/44) * np.sin(13/22 - 72 * t) + (5/37) * np.sin(2/23 - 71 * t) + (5/36) * np.sin(37/12 - 70 * t) + (5/38) * np.sin(34/45 - 69 * t) + (11/65) * np.sin(5/27 - 68 * t) + (20/61) * np.sin(25/33 - 67 * t) + (6/53) * np.sin(83/24 - 66 * t) + (9/31) * np.sin(53/33 - 65 * t) + (3/23) * np.sin(3/13 - 64 * t) + (3/13) * np.sin(8/29 - 63 * t) + (6/55) * np.sin(81/23 - 62 * t) + (2/9) * np.sin(15/11 - 61 * t) + (2/35) * np.sin(16/11 - 60 * t) + (7/20) * np.sin(59/42 - 59 * t) + (8/31) * np.sin(248/149 - 57 * t) + (2/9) * np.sin(31/38 - 56 * t) + (15/32) * np.sin(78/47 - 55 * t) + (9/55) * np.sin(230/53 - 54 * t) + (6/19) * np.sin(13/7 - 53 * t) + (8/15) * np.sin(9/5 - 51 * t) + (8/21) * np.sin(82/37 - 49 * t) + (2/11) * np.sin(185/46 - 48 * t) + (23/36) * np.sin(55/29 - 47 * t) + (11/30) * np.sin(98/47 - 45 * t) + (6/25) * np.sin(75/17 - 44 * t) + (17/19) * np.sin(200/89 - 43 * t) + (7/33) * np.sin(37/25 - 42 * t) + (8/17) * np.sin(38/15 - 41 * t) + (6/25) * np.sin(75/16 - 40 * t) + (11/13) * np.sin(122/49 - 39 * t) + (5/46) * np.sin(44/19 - 38 * t) + (23/41) * np.sin(60/23 - 37 * t) + (61/37) * np.sin(81/28 - 35 * t) + (13/24) * np.sin(43/22 - 34 * t) + (2/7) * np.sin(101/30 - 33 * t) + (35/19) * np.sin(67/23 - 31 * t) + (5/23) * np.sin(125/52 - 30 * t) + (20/21) * np.sin(46/17 - 29 * t) + (96/41) * np.sin(88/25 - 27 * t) + (15/19) * np.sin(282/113 - 26 * t) + (23/47) * np.sin(94/23 - 25 * t) + (333/98) * np.sin(123/37 - 23 * t) + (19/27) * np.sin(71/31 - 22 * t) + (13/17) * np.sin(110/27 - 21 * t) + (57/11) * np.sin(59/16 - 19 * t) + (85/31) * np.sin(54/19 - 18 * t) + (45/23) * np.sin(83/22 - 17 * t) + (131/31) * np.sin(88/25 - 15 * t) + (54/25) * np.sin(37/9 - 14 * t) + (281/54) * np.sin(221/55 - 13 * t) + (577/49) * np.sin(65/16 - 11 * t) + (2295/164) * np.sin(99/37 - 10 * t) + (91/31) * np.sin(91/24 - 9 * t) + (613/48) * np.sin(113/27 - 7 * t) + (1086/89) * np.sin(137/35 - 6 * t) + (2533/33) * np.sin(153/35 - 5 * t) + (2116/37) * np.sin(45/26 - 4 * t) + (5417/40) * np.sin(12/29 - 2 * t) + (40821/86) * np.sin(63/40 - t) - (1739/30) * np.sin(3 * t + 65/57) - (331/16) * np.sin(8 * t + 15/44) - (64/9) * np.sin(12 * t + 3/29) - (53/15) * np.sin(16 * t + 1/21) - (38/13) * np.sin(20 * t + 2/7) - (106/85) * np.sin(24 * t + 13/15) - (21/23) * np.sin(28 * t + 25/32) - (47/59) * np.sin(32 * t + 46/35) - (4/7) * np.sin(36 * t + 61/50) - (1/8) * np.sin(46 * t + 1/37) - (8/57) * np.sin(50 * t + 43/34) - (3/34) * np.sin(52 * t + 67/133) - (4/29) * np.sin(58 * t + 7/6) - (7/41) * np.sin(75 * t + 9/22) - (4/35) * np.sin(76 * t + 5/41)
        Y = (1/9 * np.sin(53/14 - 78 * t) + 3/26 * np.sin(260/71 - 77 * t) + 1/28 * np.sin(149/34 - 76 * t) + 3/16 * np.sin(155/43 - 74 * t) + 5/28 * np.sin(47/10 - 73 * t) + 5/34 * np.sin(20/19 - 72 * t) + 5/33 * np.sin(1/48 - 71 * t) + 6/29 * np.sin(156/35 - 70 * t) + 10/37 * np.sin(127/34 - 69 * t) + 1/8 * np.sin(59/71 - 68 * t) + 1/4 * np.sin(19/28 - 67 * t) + 6/31 * np.sin(173/44 - 66 * t) + 4/25 * np.sin(126/31 - 65 * t) + 1/16 * np.sin(2/23 - 64 * t) + 5/19 * np.sin(29/40 - 63 * t) + 2/21 * np.sin(83/21 - 62 * t) + 5/18 * np.sin(248/59 - 61 * t) + 3/22 * np.sin(51/29 - 60 * t) + 15/76 * np.sin(25/28 - 59 * t) + 7/33 * np.sin(57/13 - 58 * t) + 9/32 * np.sin(123/31 - 57 * t) + 5/22 * np.sin(23/38 - 56 * t) + 26/77 * np.sin(29/20 - 55 * t) + 9/26 * np.sin(205/46 - 53 * t) + 5/32 * np.sin(21/34 - 52 * t) + 10/29 * np.sin(215/143 - 51 * t) + 7/24 * np.sin(194/43 - 50 * t) + 9/26 * np.sin(164/39 - 49 * t) + 6/53 * np.sin(23/20 - 48 * t) + 3/7 * np.sin(40/21 - 47 * t) + 2/9 * np.sin(106/23 - 46 * t) + 24/49 * np.sin(41/9 - 45 * t) + 8/41 * np.sin(149/119 - 44 * t) + 17/32 * np.sin(41/24 - 43 * t) + 17/38 * np.sin(180/41 - 42 * t) + 14/25 * np.sin(136/31 - 41 * t) + 17/36 * np.sin(7/27 - 40 * t) + 7/30 * np.sin(830/277 - 39 * t) + 4/13 * np.sin(141/34 - 38 * t) + 7/20 * np.sin(11/31 - 36 * t) + 8/11 * np.sin(17/7 - 35 * t) + 44/59 * np.sin(18/31 - 32 * t) + 17/86 * np.sin(84/31 - 31 * t) + 3/5 * np.sin(95/23 - 30 * t) + 37/26 * np.sin(3/16 - 28 * t) + 24/31 * np.sin(164/47 - 27 * t) + 19/24 * np.sin(23/5 - 26 * t) + 61/33 * np.sin(23/30 - 24 * t) + 149/39 * np.sin(5/8 - 20 * t) + 29/23 * np.sin(139/39 - 19 * t) + 192/23 * np.sin(33/38 - 16 * t) + 81/17 * np.sin(55/12 - 14 * t) + 46/21 * np.sin(11/28 - 13 * t) + 537/41 * np.sin(184/183 - 12 * t) + 95/27 * np.sin(17/6 - 11 * t) + 85/24 * np.sin(31/35 - 10 * t) + 667/25 * np.sin(52/45 - 8 * t) + 137/23 * np.sin(44/41 - 7 * t) + 950/29 * np.sin(177/38 - 6 * t) + 2390/79 * np.sin(75/31 - 5 * t) + 17879/92 * np.sin(23/17 - 4 * t) + 8342/33 * np.sin(40/27 - 2 * t) + 2535/28 * np.sin(104/31 - t) - 3607/31 * np.sin(3 * t + 3/19) - 549/37 * np.sin(9 * t + 11/25) - 141/64 * np.sin(15 * t + 25/31) - 157/78 * np.sin(17 * t + 5/12) - 86/47 * np.sin(18 * t + 43/33) - 25/14 * np.sin(21 * t + 19/28) - 47/46 * np.sin(22 * t + 37/27) - 7/15 * np.sin(23 * t + 6/5) - 43/31 * np.sin(25 * t + 33/37) - 19/31 * np.sin(29 * t + 13/18) - 33/34 * np.sin(33 * t + 52/35) - 53/80 * np.sin(34 * t + 13/11) - 11/13 * np.sin(37 * t + 31/27) - 11/50 * np.sin(54 * t + 24/17) - 4/21 * np.sin(75 * t + 5/39))
    elif curvename=='cat':
        X = -(721 * np.sin(t))/4 + 196/3 * np.sin(2 * t) - 86/3 * np.sin(3 * t) - 131/2 * np.sin(4 * t) + 477/14 * np.sin(5 * t) + 27 * np.sin(6 * t) - 29/2 * np.sin(7 * t) + 68/5 * np.sin(8 * t) + 1/10 * np.sin(9 * t) + 23/4 * np.sin(10 * t) - 19/2 * np.sin(12 * t) - 85/21 * np.sin(13 * t) + 2/3 * np.sin(14 * t) + 27/5 * np.sin(15 * t) + 7/4 * np.sin(16 * t) + 17/9 * np.sin(17 * t) - 4 * np.sin(18 * t) - 1/2 * np.sin(19 * t) + 1/6 * np.sin(20 * t) + 6/7 * np.sin(21 * t) - 1/8 * np.sin(22 * t) + 1/3 * np.sin(23 * t) + 3/2 * np.sin(24 * t) + 13/5 * np.sin(25 * t) + np.sin(26 * t) - 2 * np.sin(27 * t) + 3/5 * np.sin(28 * t) - 1/5 * np.sin(29 * t) + 1/5 * np.sin(30 * t) + (2337 * np.cos(t))/8 - 43/5 * np.cos(2 * t) + 322/5 * np.cos(3 * t) - 117/5 * np.cos(4 * t) - 26/5 * np.cos(5 * t) - 23/3 * np.cos(6 * t) + 143/4 * np.cos(7 * t) - 11/4 * np.cos(8 * t) - 31/3 * np.cos(9 * t) - 13/4 * np.cos(10 * t) - 9/2 * np.cos(11 * t) + 41/20 * np.cos(12 * t) + 8 * np.cos(13 * t) + 2/3 * np.cos(14 * t) + 6 * np.cos(15 * t) + 17/4 * np.cos(16 * t) - 3/2 * np.cos(17 * t) - 29/10 * np.cos(18 * t) + 11/6 * np.cos(19 * t) + 12/5 * np.cos(20 * t) + 3/2 * np.cos(21 * t) + 11/12 * np.cos(22 * t) - 4/5 * np.cos(23 * t) + np.cos(24 * t) + 17/8 * np.cos(25 * t) - 7/2 * np.cos(26 * t) - 5/6 * np.cos(27 * t) - 11/10 * np.cos(28 * t) + 1/2 * np.cos(29 * t) - 1/5 * np.cos(30 * t)
        Y = -(637 * np.sin(t))/2 - 188/5 * np.sin(2 * t) - 11/7 * np.sin(3 * t) - 12/5 * np.sin(4 * t) + 11/3 * np.sin(5 * t) - 37/4 * np.sin(6 * t) + 8/3 * np.sin(7 * t) + 65/6 * np.sin(8 * t) - 32/5 * np.sin(9 * t) - 41/4 * np.sin(10 * t) - 38/3 * np.sin(11 * t) - 47/8 * np.sin(12 * t) + 5/4 * np.sin(13 * t) - 41/7 * np.sin(14 * t) - 7/3 * np.sin(15 * t) - 13/7 * np.sin(16 * t) + 17/4 * np.sin(17 * t) - 9/4 * np.sin(18 * t) + 8/9 * np.sin(19 * t) + 3/5 * np.sin(20 * t) - 2/5 * np.sin(21 * t) + 4/3 * np.sin(22 * t) + 1/3 * np.sin(23 * t) + 3/5 * np.sin(24 * t) - 3/5 * np.sin(25 * t) + 6/5 * np.sin(26 * t) - 1/5 * np.sin(27 * t) + 10/9 * np.sin(28 * t) + 1/3 * np.sin(29 * t) - 3/4 * np.sin(30 * t) - (125 * np.cos(t))/2 - 521/9 * np.cos(2 * t) - 359/3 * np.cos(3 * t) + 47/3 * np.cos(4 * t) - 33/2 * np.cos(5 * t) - 5/4 * np.cos(6 * t) + 31/8 * np.cos(7 * t) + 9/10 * np.cos(8 * t) - 119/4 * np.cos(9 * t) - 17/2 * np.cos(10 * t) + 22/3 * np.cos(11 * t) + 15/4 * np.cos(12 * t) - 5/2 * np.cos(13 * t) + 19/6 * np.cos(14 * t) + 7/4 * np.cos(15 * t) + 31/4 * np.cos(16 * t) - np.cos(17 * t) + 11/10 * np.cos(18 * t) - 2/3 * np.cos(19 * t) + 13/3 * np.cos(20 * t) - 5/4 * np.cos(21 * t) + 2/3 * np.cos(22 * t) + 1/4 * np.cos(23 * t) + 5/6 * np.cos(24 * t) + 3/4 * np.cos(26 * t) - 1/2 * np.cos(27 * t) - 1/10 * np.cos(28 * t) - 1/3 * np.cos(29 * t) - 1/19 * np.cos(30 * t)
    elif curvename=='dolphin':
        X = 4/23*np.sin(62/33-58*t) + 8/11*np.sin(10/9-56*t) + 17/24*np.sin(38/35-55*t) + 30/89*np.sin(81/23-54*t) + 3/17*np.sin(53/18-53*t) + 21/38*np.sin(29/19-52*t) + 11/35*np.sin(103/40-51*t) + 7/16*np.sin(79/18-50*t) + 4/15*np.sin(270/77-49*t) + 19/35*np.sin(59/27-48*t) + 37/43*np.sin(71/17-47*t) + np.sin(18/43-45*t) + 21/26*np.sin(37/26-44*t) + 27/19*np.sin(111/32-42*t) + 8/39*np.sin(13/25-41*t) + 23/30*np.sin(27/8-40*t) + 23/21*np.sin(32/35-37*t) + 18/37*np.sin(91/31-36*t) + 45/22*np.sin(29/37-35*t) + 56/45*np.sin(11/8-33*t) + 4/7*np.sin(32/19-32*t) + 54/23*np.sin(74/29-31*t) + 28/19*np.sin(125/33-30*t) + 19/9*np.sin(73/27-29*t) + 16/17*np.sin(737/736-28*t) + 52/33*np.sin(130/29-27*t) + 41/23*np.sin(43/30-25*t) + 29/20*np.sin(67/26-24*t) + 64/25*np.sin(136/29-23*t) + 162/37*np.sin(59/34-21*t) + 871/435*np.sin(199/51-20*t) + 61/42*np.sin(58/17-19*t) + 159/25*np.sin(77/31-17*t) + 241/15*np.sin(94/31-13*t) + 259/18*np.sin(114/91-12*t) + 356/57*np.sin(23/25-11*t) + 2283/137*np.sin(23/25-10*t) + 1267/45*np.sin(139/42-9*t) + 613/26*np.sin(41/23-8*t) + 189/16*np.sin(122/47-6*t) + 385/6*np.sin(151/41-5*t) + 2551/38*np.sin(106/35-4*t) + 1997/18*np.sin(6/5-2*t) + 43357/47*np.sin(81/26-t) - 4699/35*np.sin(3*t+25/31) - 1029/34*np.sin(7*t+20/21) - 250/17*np.sin(14*t+7/40) - 140/17*np.sin(15*t+14/25) - 194/29*np.sin(16*t+29/44) - 277/52*np.sin(18*t+37/53) - 94/41*np.sin(22*t+33/31) - 57/28*np.sin(26*t+44/45) - 128/61*np.sin(34*t+11/14) - 111/95*np.sin(38*t+55/37) - 85/71*np.sin(39*t+4/45) - 25/29*np.sin(43*t+129/103) - 7/37*np.sin(46*t+9/20) - 17/32*np.sin(57*t+11/28) - 5/16*np.sin(59*t+32/39)
        Y = 5/11*np.sin(163/37-59*t) + 7/22*np.sin(19/41-58*t) + 30/41*np.sin(1-57*t) + 37/29*np.sin(137/57-56*t) + 5/7*np.sin(17/6-55*t) + 11/39*np.sin(46/45-52*t) + 25/28*np.sin(116/83-51*t) + 25/34*np.sin(11/20-47*t) + 8/27*np.sin(81/41-46*t) + 44/39*np.sin(78/37-45*t) + 11/25*np.sin(107/37-44*t) + 7/20*np.sin(7/16-41*t) + 30/31*np.sin(19/5-40*t) + 37/27*np.sin(148/59-39*t) + 44/39*np.sin(17/27-38*t) + 13/11*np.sin(7/11-37*t) + 28/33*np.sin(119/39-36*t) + 27/13*np.sin(244/81-35*t) + 13/23*np.sin(113/27-34*t) + 47/38*np.sin(127/32-33*t) + 155/59*np.sin(173/45-29*t) + 105/37*np.sin(22/43-27*t) + 106/27*np.sin(23/37-26*t) + 97/41*np.sin(53/29-25*t) + 83/45*np.sin(109/31-24*t) + 81/31*np.sin(96/29-23*t) + 56/37*np.sin(29/10-22*t) + 44/13*np.sin(29/19-19*t) + 18/5*np.sin(34/31-18*t) + 163/51*np.sin(75/17-17*t) + 152/31*np.sin(61/18-16*t) + 146/19*np.sin(47/20-15*t) + 353/35*np.sin(55/48-14*t) + 355/28*np.sin(102/25-12*t) + 1259/63*np.sin(71/18-11*t) + 17/35*np.sin(125/52-10*t) + 786/23*np.sin(23/26-6*t) + 2470/41*np.sin(77/30-5*t) + 2329/47*np.sin(47/21-4*t) + 2527/33*np.sin(23/14-3*t) + 9931/33*np.sin(51/35-2*t) - 11506/19*np.sin(t+56/67) - 2081/42*np.sin(7*t+9/28) - 537/14*np.sin(8*t+3/25) - 278/29*np.sin(9*t+23/33) - 107/15*np.sin(13*t+35/26) - 56/19*np.sin(20*t+5/9) - 5/9*np.sin(21*t+1/34) - 17/24*np.sin(28*t+36/23) - 21/11*np.sin(30*t+27/37) - 138/83*np.sin(31*t+1/7) - 10/17*np.sin(32*t+29/48) - 31/63*np.sin(42*t+27/28) - 4/27*np.sin(43*t+29/43) - 13/24*np.sin(48*t+5/21) - 4/7*np.sin(49*t+29/23) - 26/77*np.sin(50*t+29/27) - 19/14*np.sin(53*t+61/48) + 34/25*np.sin(54*t+37/26)
    elif curvename=='giraffe':
        X = (1/7)*np.sin(21/8 - 120*t) + (1/11)*np.sin(10/11 - 119*t) + (1/33)*np.sin(28/9 - 117*t) + (1/8)*np.sin(13/10 - 115*t) + (2/9)*np.sin(3 - 113*t) + (1/11)*np.sin(22/5 - 112*t) + (1/5)*np.sin(43/11 - 111*t) + (2/11)*np.sin(2/9 - 110*t) + (1/18)*np.sin(8/5 - 109*t) + (1/7)*np.sin(12/13 - 108*t) + (1/15)*np.sin(1/81 - 107*t) + (1/5)*np.sin(1/3 - 106*t) + (1/8)*np.sin(11/6 - 105*t) + (1/12)*np.sin(1/3 - 104*t) + (1/11)*np.sin(21/5 - 103*t) + (1/23)*np.sin(57/29 - 101*t) + (1/7)*np.sin(3/2 - 100*t) + (1/49)*np.sin(11/16 - 99*t) + (1/12)*np.sin(29/13 - 96*t) + (1/5)*np.sin(47/19 - 95*t) + (1/10)*np.sin(27/10 - 94*t) + (1/35)*np.sin(9/8 - 93*t) + (4/13)*np.sin(19/11 - 92*t) + (1/3)*np.sin(11/10 - 91*t) + (1/4)*np.sin(7/8 - 89*t) + (1/9)*np.sin(25/6 - 88*t) + (1/7)*np.sin(220/73 - 87*t) + (2/11)*np.sin(33/16 - 85*t) + (1/4)*np.sin(19/13 - 84*t) + (3/10)*np.sin(32/11 - 83*t) + (2/9)*np.sin(13/11 - 82*t) + (1/9)*np.sin(32/9 - 81*t) + (7/11)*np.sin(5/14 - 80*t) + (1/25)*np.sin(4/3 - 79*t) + (3/10)*np.sin(33/14 - 78*t) + (5/9)*np.sin(52/15 - 77*t) + (4/11)*np.sin(28/17 - 76*t) + (1/5)*np.sin(9/8 - 74*t) + (1/3)*np.sin(4/13 - 73*t) + (1/8)*np.sin(17/16 - 72*t) + (2/13)*np.sin(3/5 - 71*t) + (4/13)*np.sin(59/14 - 70*t) + (1/3)*np.sin(1/3 - 69*t) + (2/5)*np.sin(24/25 - 68*t) + (3/4)*np.sin(57/23 - 67*t) + (3/7)*np.sin(29/9 - 66*t) + (1/9)*np.sin(20/7 - 65*t) + (3/10)*np.sin(5/2 - 63*t) + (3/5)*np.sin(12/5 - 61*t) + (20/19)*np.sin(7/2 - 59*t) + (31/32)*np.sin(7/12 - 58*t) + (8/11)*np.sin(24/11 - 57*t) + (5/8)*np.sin(11/9 - 56*t) + (2/3)*np.sin(27/7 - 55*t) + (3/7)*np.sin(2/11 - 54*t) + (1/6)*np.sin(8/11 - 52*t) + (1/16)*np.sin(15/4 - 51*t) + (7/6)*np.sin(38/13 - 50*t) + (29/28)*np.sin(73/18 - 49*t) + (3/7)*np.sin(15/4 - 48*t) + (16/15)*np.sin(13/9 - 47*t) + (7/12)*np.sin(17/5 - 46*t) + (3/4)*np.sin(19/9 - 45*t) + (17/11)*np.sin(43/16 - 43*t) + (7/8)*np.sin(41/9 - 42*t) + (16/11)*np.sin(20/9 - 41*t) + (7/6)*np.sin(3/2 - 40*t) + (17/10)*np.sin(23/7 - 39*t) + (9/11)*np.sin(33/8 - 35*t) + (17/12)*np.sin(4 - 34*t) + (24/23)*np.sin(83/21 - 33*t) + (23/10)*np.sin(43/13 - 32*t) + (11/8)*np.sin(26/7 - 31*t) + (28/13)*np.sin(25/12 - 30*t) + (27/11)*np.sin(33/13 - 29*t) + (13/3)*np.sin(51/11 - 28*t) + (11/10)*np.sin(31/7 - 27*t) + (25/13)*np.sin(43/17 - 25*t) + (3/4)*np.sin(112/37 - 24*t) + (37/10)*np.sin(9/4 - 23*t) + (23/9)*np.sin(16/9 - 22*t) + (21/8)*np.sin(37/9 - 21*t) + (11/3)*np.sin(17/8 - 19*t) + (33/8)*np.sin(67/15 - 18*t) + (10/3)*np.sin(47/10 - 17*t) + (69/11)*np.sin(30/7 - 16*t) + (43/13)*np.sin(17/11 - 15*t) + (19/3)*np.sin(171/43 - 14*t) + (60/7)*np.sin(22/13 - 13*t) + (62/13)*np.sin(37/9 - 12*t) + (17/4)*np.sin(25/9 - 11*t) + (127/6)*np.sin(37/8 - 10*t) + (1/3)*np.sin(11/10 - 9*t) + (187/9)*np.sin(77/17 - 8*t) + (31/4)*np.sin(29/8 - 7*t) + (205/14)*np.sin(25/6 - 5*t) + (34/3)*np.sin(37/16 - 4*t) + (259/12)*np.sin(2/7 - 3*t) + (892/27)*np.sin(23/5 - 2*t) + (1367/7)*np.sin(23/9 - t) - (389/13)*np.sin(6*t + 1/4) - (23/8)*np.sin(20*t + 2/3) - (11/5)*np.sin(26*t + 16/13) - (7/5)*np.sin(36*t + 2/3) - (8/25)*np.sin(37*t + 1/3) - (10/9)*np.sin(38*t + 2/9) - (9/7)*np.sin(44*t + 11/8) - (2/7)*np.sin(53*t + 4/5) - (1/4)*np.sin(60*t + 7/11) - (4/9)*np.sin(62*t + 1/3) - (2/3)*np.sin(64*t + 4/9) - (2/9)*np.sin(75*t + 33/34) - (3/8)*np.sin(86*t + 1) - (1/7)*np.sin(90*t + 5/12) - (1/24)*np.sin(97*t + 1/9) - (1/7)*np.sin(98*t + 5/9) - (1/11)*np.sin(102*t + 1/11) - (1/5)*np.sin(114*t + 4/11) - (1/48)*np.sin(116*t + 15/16) - (1/7)*np.sin(118*t + 4/5) - 26/5
        Y = 1/18 * np.sin(13/4 - 120 * t) + 1/6 * np.sin(45/11 - 119 * t) + 1/23 * np.sin(17/16 - 118 * t) + 1/13 * np.sin(20/7 - 117 * t) + 1/6 * np.sin(15/7 - 116 * t) + 1/13 * np.sin(3/7 - 115 * t) + 2/11 * np.sin(3/5 - 114 * t) + 1/9 * np.sin(2/5 - 113 * t) + 1/10 * np.sin(34/15 - 112 * t) + 1/15 * np.sin(13/7 - 111 * t) + 1/20 * np.sin(8/5 - 110 * t) + 1/5 * np.sin(15/4 - 108 * t)  + 1/25 * np.sin(26/11 - 107 * t) + 1/6 * np.sin(40/17 - 103 * t) + 1/14 * np.sin(18/5 - 102 * t)  + 1/9 * np.sin(41/20 - 101 * t) + 1/7 * np.sin(2 - 100 * t) + 5/16 * np.sin(13/5 - 99 * t)  + 1/12 * np.sin(11/3 - 98 * t) + 1/5 * np.sin(1/2 - 96 * t) + 1/67 * np.sin(15/16 - 95 * t)  + 1/6 * np.sin(68/15 - 93 * t) + 1/6 * np.sin(25/13 - 92 * t) + 1/21 * np.sin(3/7 - 91 * t)  + 1/13 * np.sin(24/7 - 90 * t) + 2/9 * np.sin(31/8 - 89 * t) + 3/13 * np.sin(49/15 - 88 * t)  + 1/8 * np.sin(3/2 - 87 * t) + 1/11 * np.sin(18/11 - 86 * t) + 2/7 * np.sin(45/11 - 85 * t)  + 1/6 * np.sin(11/3 - 84 * t) + 1/9 * np.sin(20/11 - 83 * t) + 4/11 * np.sin(33/16 - 82 * t)  + 1/8 * np.sin(45/11 - 81 * t) + 1/18 * np.sin(7/6 - 80 * t) + 2/11 * np.sin(20/9 - 78 * t)  + 1/10 * np.sin(24/11 - 76 * t) + 2/5 * np.sin(45/13 - 74 * t) + 1/7 * np.sin(81/23 - 73 * t)  + 3/7 * np.sin(23/7 - 72 * t) + 4/9 * np.sin(25/11 - 69 * t) + 1/5 * np.sin(43/16 - 68 * t)  + 2/13 * np.sin(17/7 - 67 * t) + 3/4 * np.sin(25/9 - 65 * t) + 5/14 * np.sin(2 - 64 * t)  + 5/11 * np.sin(18/5 - 63 * t) + 4/7 * np.sin(1/14 - 62 * t) + 8/13 * np.sin(47/11 - 61 * t)  + 4/11 * np.sin(1/6 - 60 * t) + 12/23 * np.sin(6/7 - 58 * t) + 1/2 * np.sin(19/11 - 56 * t)  + 7/12 * np.sin(41/10 - 55 * t) + 7/10 * np.sin(53/18 - 54 * t) + 5/14 * np.sin(18/7 - 53 * t)  + 2/3 * np.sin(21/16 - 52 * t) + 11/15 * np.sin(38/9 - 51 * t) + 3/5 * np.sin(29/7 - 50 * t)  + 10/7 * np.sin(9/8 - 48 * t) + 1/4 * np.sin(3/7 - 47 * t) + 2/5 * np.sin(47/14 - 46 * t)  + 4/11 * np.sin(85/19 - 45 * t) + 5/8 * np.sin(21/16 - 44 * t) + 3/4 * np.sin(35/8 - 42 * t)  + 7/11 * np.sin(31/7 - 40 * t) + 5/9 * np.sin(51/11 - 39 * t) + 1/3 * np.sin(25/6 - 38 * t)  + 10/19 * np.sin(29/19 - 37 * t) + 7/6 * np.sin(29/11 - 35 * t) + 8/9 * np.sin(13/6 - 34 * t)  + 18/17 * np.sin(38/11 - 33 * t) + 53/21 * np.sin(43/22 - 30 * t) + 9/11 * np.sin(49/16 - 29 * t)  + 2/3 * np.sin(17/7 - 26 * t) + 3/2 * np.sin(25/7 - 25 * t) + 37/36 * np.sin(14/3 - 24 * t)  + 28/11 * np.sin(18/5 - 22 * t) + 1/2 * np.sin(16/5 - 21 * t) + 9/10 * np.sin(19/9 - 20 * t)  + 29/11 * np.sin(16/7 - 19 * t) + 12/7 * np.sin(67/34 - 18 * t) + 25/8 * np.sin(51/11 - 17 * t)  + 30/7 * np.sin(209/52 - 16 * t) + 48/11 * np.sin(29/7 - 15 * t) + 63/10 * np.sin(37/36 - 14 * t)  + 193/17 * np.sin(7/11 - 13 * t) + 31/12 * np.sin(47/10 - 12 * t) + 172/9 * np.sin(65/14 - 11 * t)  + 531/19 * np.sin(3/8 - 9 * t) + 369/13 * np.sin(111/28 - 8 * t) + 752/9 * np.sin(63/16 - 7 * t)  + 1106/27 * np.sin(63/16 - 6 * t) + 127/6 * np.sin(1/12 - 4 * t) + 3281/15 * np.sin(7/6 - 2 * t)  + 1846/5 * np.sin(67/15 - t) - 482/37 * np.sin(10 * t) - 1325/12 * np.sin(3 * t + 9/7)  - 97/15 * np.sin(5 * t + 20/13) - 33/10 * np.sin(23 * t + 10/7) - 34/9 * np.sin(27 * t + 5/6)  - 11/4 * np.sin(28 * t + 4/11) - 8/7 * np.sin(31 * t + 17/12) - 17/7 * np.sin(32 * t + 4/11)  - 3/5 * np.sin(36 * t + 1/5) - 8/9 * np.sin(41 * t + 31/21) - 3/5 * np.sin(43 * t + 12/11)  - 3/8 * np.sin(49 * t + 7/6) - 3/10 * np.sin(57 * t + 20/13) - 5/7 * np.sin(59 * t + 11/7)  - 2/5 * np.sin(66 * t + 1/4) - 1/3 * np.sin(70 * t + 15/11) - 1/5 * np.sin(71 * t + 1/9)  - 1/3 * np.sin(75 * t + 7/9) - 1/5 * np.sin(77 * t + 10/7) - 1/5 * np.sin(79 * t + 1/33)  - 2/11 * np.sin(94 * t + 8/17) - 1/21 * np.sin(104 * t + 12/13) - 1/7 * np.sin(105 * t + 1/48)  - 1/10 * np.sin(106 * t + 34/33) - 1/10 * np.sin(109 * t + 8/7) - 1721/12
    else :
        print("The name given to the entry does not correspond to a recorded curve. A circle curve will be returned.")
        X = np.cos(t) 
        Y = np.sin(t)

    Target = np.array([X,Y]).T

    return Target

def simulation_Cucker_Smale(N, alpha, beta, K, M, T, h, adjacency_psi, adjacency_phi, Z, Xinit=None, Vinit=None, random_seed=None):
    """
    Simulate the Cucker-Smale model with pattern formation.

    Parameters:
    - N: int, number of agents.
    - alpha: float, parameter for the singular kernel.
    - beta: float, parameter for the potential energy function.
    - K: float, coupling strength for velocity alignment.
    - M: float, coupling strength for formation control.
    - T: int, number of timesteps.
    - h: float, time step size.
    - adjacency_psi: adjacency matrix for velocity alignment interactions (N x N).
    - adjacency_phi: adjacency matrix for formation control interactions (N x N).
    - Z: array of shape (N, 2), target positions for pattern formation.
    - Xinit: array of shape (N, 2), initial positions of agents (optional).
    - Vinit: array of shape (N, 2), initial velocities of agents (optional).
    - random_seed: int, random seed for reproducibility (optional).

    Returns:
    - X: array of shape (T, N, 2), positions of agents over time.
    - V: array of shape (T, N, 2), velocities of agents over time.
    """
    # Set up a seed to replicate the simulation (if needed) 
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize positions (with normalization) 
    if Xinit is None:
        Xinitlen = int(np.max(Z) - np.min(Z))
        coeff = 1 # can be adapted ; < 1: initialized in a smaller zone than the final zone; > 1: initialized in a larger zone than the final zone
        X_range = Xinitlen * coeff
        Xinit = np.random.rand(N, 2) * X_range - X_range / 2
        Xinit -= np.mean(Xinit, axis=0)
        Xinit += np.mean(Z, axis=0)

    # Intialize velocities (with normalization)
    if Vinit is None:
        v_range = 50 # the higher it is, the greater the initial velocity of the particles
        Vinit = np.random.rand(N, 2) * v_range - v_range / 2
        Vinit -= np.mean(Vinit, axis=0)

    # Initialize arrays to store positions and velocities
    X = np.array([Xinit])
    V = np.array([Vinit])

    # Simulation loop
    for t in range(1, T+1):
        Xnow = np.copy(X[t - 1])
        Vnow = np.copy(V[t - 1])

        # Runge-Kutta integration
        K1 = np.array([
            Vnow,
            cucker_smale_model_pattern_formation(Xnow, Vnow, Z, adjacency_psi, adjacency_phi, alpha, beta, K, M, N)
        ])
        K2 = np.array([
            Vnow + K1[1] * h / 2,
            cucker_smale_model_pattern_formation(Xnow + K1[0] * h / 2, Vnow + K1[1] * h / 2, Z, adjacency_psi, adjacency_phi, alpha, beta, K, M, N)
        ])
        K3 = np.array([
            Vnow + K2[1] * h / 2,
            cucker_smale_model_pattern_formation(Xnow + K2[0] * h / 2, Vnow + K2[1] * h / 2, Z, adjacency_psi, adjacency_phi, alpha, beta, K, M, N)
        ])
        K4 = np.array([
            Vnow + K3[1] * h,
            cucker_smale_model_pattern_formation(Xnow + K3[0] * h, Vnow + K3[1] * h, Z, adjacency_psi, adjacency_phi, alpha, beta, K, M, N)
        ])

        Xnext = np.copy(Xnow)
        Vnext = np.copy(Vnow)

        Xnext += (K1[0] + 2 * K2[0] + 2 * K3[0] + K4[0]) * h / 6
        Vnext += (K1[1] + 2 * K2[1] + 2 * K3[1] + K4[1]) * h / 6

        # Ensure no NaN values
        Xnext = np.nan_to_num(Xnext)
        Vnext = np.nan_to_num(Vnext)

        # Append results
        X = np.append(X, np.array([Xnext]), axis=0)
        V = np.append(V, np.array([Vnext]), axis=0)

    return X, V

if __name__ == "__main__":

    
    # Parameters configuration for pi curve
    N = 30
    alpha = 1.0
    beta = 0.5
    K = 450
    M = 150
    T = 600
    h = 0.025 # de base 0.025
    adjacency_psi = adjacency_matrix("0", N)
    adjacency_phi = adjacency_matrix("1", N)
    Z = pattern_formation('pi', N) 

    # Simulation

    X, V = simulation_Cucker_Smale(N, alpha, beta, K, M, T, h, adjacency_psi, adjacency_phi, Z, random_seed=3)

    # Plots or animations

    #makeplot_animation_only_agents(X,V,N,T,h,Z,SAVEFIG=True,filename_template="animation_only_agents_pi_cruve")
    #make_plot_snapshot_energy(X, V, adjacency_phi, beta, h, Z, M, SAVEFIG=True,filename_template="snapchot_energy_pi_curve")
    make_plot_snapshot_only_agents(X, V, N, [0, 120, 240, 360, 480, 600], h, Z, SAVEFIG=True, filename_template="snapshot_only_agents_pi_curve")

    # Parameters configuration for omega curve
    N = 30
    alpha = 1.0
    beta = 0.5
    K = 450
    M = 150
    T = 600
    h = 0.025 # de base 0.025
    adjacency_psi = adjacency_matrix("0", N)
    adjacency_phi = adjacency_matrix("1", N)
    Z = pattern_formation('omega', N) 

    # Simulation

    X, V = simulation_Cucker_Smale(N, alpha, beta, K, M, T, h, adjacency_psi, adjacency_phi, Z, random_seed=3)

    # Plots or animations

    #makeplot_animation_only_agents(X,V,N,T,h,Z,SAVEFIG=True,filename_template="animation_only_agents_omega_cruve")
    #make_plot_snapshot_energy(X, V, adjacency_phi, beta, h, Z, M, SAVEFIG=True,filename_template="snapchot_energy_omega_curve")
    make_plot_snapshot_only_agents(X, V, N, [0, 120, 240, 360, 480, 600], h, Z, SAVEFIG=True, filename_template="snapshot_only_agents_omega_curve")
