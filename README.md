# SpaceRobotics7863

This repository contains projects and coursework for ROB-GY 7863: Space Robotics at the NYU Tandon School of Engineering.

## Project 1: Simulating In-Space Disassembly Tasks in a Zero-G Environment

### Project Overview

This project focuses on the development of a high-fidelity simulation environment for in-space robotic disassembly tasks, specifically simulating a 7-axis KUKA IIWA manipulator performing an unscrewing operation on a free-floating object in a microgravity environment.

The simulation is built using the MuJoCo physics engine and leverages models from Google's MuJoCo Menagerie. The primary goal is to create a realistic and robust platform for testing and validating autonomous robotic procedures for On-orbit Servicing, Assembly, and Manufacturing (OSAM) missions.

### Key Features

- **Robotic Arm:** A 7-DOF KUKA IIWA redundant manipulator.
- **Environment:** A simulated microgravity environment ($10^{-6} g$) designed to mimic conditions inside the International Space Station (ISS). The visual environment is based on the US Lab module of the ISS.
- **Task:** The robot interacts with a free-floating box, loaded from a custom STL file, to perform a precise manipulation task.
- **Control:** A Jacobian Damped Least Squares (DLS) controller is implemented to provide robust control and gracefully handle kinematic singularities.
- **Collision Handling:** The workspace is enclosed by simple bounding walls for efficient collision detection, while complex ISS geometry is used for visualization only.

### How to Run the Simulation

1.  **Navigate to the project directory:**
    ```bash
    cd Project1/
    ```

2.  **Ensure you have the required dependencies installed:**
    - `mujoco`
    - `numpy`
    - `matplotlib`
    - `scipy`

3.  **Run the main simulation file:**
    ```bash
    python integrated_simulation.py
    ```
    You can run more parameters using command line arguments such as `--duration` or `--plot-only`.

### Repository Structure

- `Project1/`: Contains all the files related to this simulation project.
  - `integrated_simulation.py`: The main Python script to run the simulation.
  - `*.xml`: MuJoCo model files defining the robot, objects, and environment.
  - `*.stl`: STL files for custom geometries like the target box.
  - `Report.tex` / `ref.bib`: The LaTeX source for the project report and its citations.
- `mujoco_menagerie/`: Contains the original robot models from the Google MuJoCo Menagerie.
