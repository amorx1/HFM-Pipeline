# HFM-Pipeline
Complete Python pipeline for:
  1. Extracting spatio-temporal data from 2D/3D Hidden Fluid Mechanics simulations (.vtu data)
  2. Creating, training and testing "Physics-Informed" convolutional neural network, with manual hyperparameter tuning
  3. Visualizing regressed spatio-temporal mesh data

The program is a solution for the extensive data manipulation that needed to be done when moving data to and from a neural network pipeline, as part of a research project. The data was provided in the form of 2D simulations, from which spatial and temporal coordinates, and concentration, velocity and pressure field data had to be retrieved and exported to csv files. These then had to be iterated through to extract and append the data to another csv file for each timestep, before converting everything again to a .mat file as the input to the the neural network.

The neural network was configured to output predictions in the form of .mat files, meaning these had to be converted back into csv and then vtu files in order to visualize the results. This could be done in MATLAB, however, Python > MATLAB.
