# Generalized Cucker-Smale Model Simulation

This repository contains the implementation of a simulation framework for the generalized Cucker-Smale model, written by Antoine S.. The code is based on the article *"A Collisionless Singular Cucker-Smale Model with Decentralized Formation Control"* by Young-Pil Choi, Dante Kalise, Jan Peszek, and Andrés A. Peters, and was developed as part of a study to analyze the model under various configurations and parameter settings.

**Note:** The code was written for educational and experimental purposes and is not optimized for computational efficiency.

## Repository Structure

The repository is organized as follows:

- **`Basic_model.py`**: Contains the implementation of the baseline model and serves as the foundation for the simulations.
- **`Network_structures.py`**: Includes the definitions and implementation of various network configurations used in the analysis.
- **`Parameters_Analysis.py`**: Handles the parameter study, exploring the influence of key parameters on the model's behavior.
- **`visualization.py`**: Provides utilities for visualizing the simulation results, such as agent trajectories and energy dissipation curves.
- **`visualization_graph_type_and_curve.py`**: Includes functions for generating differents types of curves.

## Figures and Animations

All figures and animations generated during the study are stored in their respective folders under the `figures` directory. These include results corresponding to each section of the report, as well as animations of the agents' movements over time.
