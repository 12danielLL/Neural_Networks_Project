# Neural Networks Project

This repository contains the final project for the Neural Networks course . The project involves analyzing neural activity data to classify neuron types based on their electrical responses. The dataset was provided by the Allen Institute for Brain Science, with neurons sampled from human and mouse cortices.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Project Overview](#project-overview)
   - [Goals](#goals)
   - [Tasks](#tasks)
3. [Key Results](#key-results)
4. [Visualizations](#visualizations)
5. [How to Run](#how-to-run)
   - [Prerequisites](#prerequisites)
   - [Steps](#steps)
   - [Troubleshooting](#troubleshooting)
6. [File Structure](#file-structure)
7. [Technologies](#technologies)
8. [License](#license)

## Project Overview

### Goals
1. Analyze and visualize neural activity data.
2. Train classifiers to distinguish between neuron types: `spiny` and `aspiny`.
3. Explore unsupervised learning methods like PCA and Autoencoders for data dimensionality reduction.
4. Implement and evaluate supervised learning models, including Logistic Regression, Perceptron, and MLP.

### Tasks
The project is divided into three main parts:

#### Part A: Data Analysis and Visualization
1. Description of `spiny` and `aspiny` neuron types.
2. Visualization of electrical stimuli over time.
3. Analysis of neuron responses to stimuli.
4. Calculation of spike trains and time-dependent firing rates.

#### Part B: Unsupervised Learning
1. Principal Component Analysis (PCA) for dimensionality reduction.
2. Implementation of an Autoencoder to compress and reconstruct the data.

#### Part C: Supervised Learning
1. Training and evaluation of Logistic Regression and Perceptron models.
2. Training a Multi-Layer Perceptron (MLP) with Gradient Descent (GD) and Stochastic Gradient Descent (SGD).

## Key Results
- **PCA**: Identified 27 principal components needed to retain 99% of the data's variance.
- **Autoencoder**: Achieved a mean squared error (MSE) of 4% after 50 epochs, but PCA provided better separation of neuron types.
- **Logistic Regression and Perceptron**:
  - Logistic Regression outperformed Perceptron with a higher recall (0.99 vs. 0.94).
- **MLP**:
  - GD and SGD models achieved similar performance (precision and recall = 0.99).
  - No significant overfitting observed.

## Visualizations
The `Results.pdf` document contains graphs and charts for:
- Firing rates of neurons.
- PCA projections in 2D and 3D.
- Confusion matrices and ROC curves for classifiers.
- 
### How to Run
## Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- pip (Python package manager)

1. Clone this repository to your local machine:
```bash
git clone https://github.com/12danielLL/Neural-Networks-Project.git
```

2. Navigate to the project directory:
```bash
cd Neural-Networks-Project
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Open the Jupyter Notebook:
```bash
jupyter notebook Neural_Networks_Project.ipynb
```

5. Execute the cells in sequence to reproduce the analysis, visualizations, and results.

## Troubleshooting
- If you encounter missing libraries, install them individually using:
```bash
pip install <name of library>
```
- For any other issues or questions, please open an issue on GitHub.

## File Structure

```plaintext
├── LICENSE                                # License file for the project
├── Neural Networks Project - Instructions.pdf # Detailed project instructions
├── Neural Networks Project - Results.pdf      # Report with project results and visualizations
├── Neural_Networks_Project.ipynb              # Jupyter Notebook with code implementation
├── README.md                            # Documentation for the project
```
## Technologies
- Python
- TensorFlow/PyTorch
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib

# License
This project is distributed under the MIT License. See `LICENSE` file for more details.

