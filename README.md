# Statistical Machine Learning for Astronomy: Tutorial Repository

This repository contains companion tutorials for the textbook **[Statistical Machine Learning for Astronomy](https://arxiv.org/abs/2506.12230)** by Yuan-Sen Ting.

## Overview

These tutorials provide hands-on, practical implementations of the concepts covered in the textbook. Each tutorial is designed as a standalone Jupyter notebook that explores key statistical and machine learning concepts through real astronomical applications.

## Tutorial Status

We are currently in the process of cleaning up and standardizing all tutorials from the textbook. The tutorials listed below have been fully revised and are ready for use. Additional tutorials will be added as they are completed.

## Available Tutorials

### Chapter 2a: Probabilistic Distribution
- **File**: `tutorial_chapter2.ipynb`
- **Topics**: Probability distributions, Poisson processes, maximum likelihood estimation, spatial statistics
- **Application**: Detecting stellar clusters through statistical analysis of spatial point patterns

### Chapter 3: Statistical Foundations and Summary Statistics
- **File**: `tutorial_chapter3.ipynb`
- **Topics**: Statistical moments, correlation functions, bootstrap methods, two-point statistics
- **Application**: Detecting the Baryon Acoustic Oscillation signal in simulated cosmological data

### Chapter 7: Classification and Logistic Regression
- **File**: `tutorial_chapter7.ipynb`
- **Topics**: Logistic regression, sigmoid transformation, gradient descent optimization, classification metrics, hyperparameter tuning
- **Application**: Distinguishing Red Clump from Red Giant Branch stars using stellar parameters from APOGEE

### Chapter 9: Bayesian Logistic Regression
- **File**: `tutorial_chapter9.ipynb`
- **Topics**: Bayesian inference for classification, Laplace approximation, predictive distributions, uncertainty quantification, prior specification
- **Application**: Quantifying classification uncertainty for Red Clump vs Red Giant Branch stars with parameter uncertainty propagation

### Chapter 11: K-means and Gaussian Mixture Models
- **File**: `tutorial_chapter11.ipynb`
- **Topics**: K-means clustering, K-means++ initialization, Gaussian Mixture Models, Expectation-Maximization algorithm, model selection with AIC/BIC
- **Application**: Identifying stellar populations in the Sculptor dwarf galaxy through chemical abundance clustering to reveal episodic star formation history

### Chapter 12: Sampling and Monte Carlo Methods
- **File**: `tutorial_chapter12.ipynb`
- **Topics**: Inverse transform sampling, rejection sampling, importance sampling, Monte Carlo integration, effective sample size
- **Application**: Generating realistic stellar populations using the Kroupa Initial Mass Function and estimating population properties

## Prerequisites

To run these tutorials, you'll need:
- Python 3.7 or higher
- NumPy
- Matplotlib
- SciPy
- Jupyter Notebook or JupyterLab

## Usage

Each tutorial is self-contained and includes:
- Theoretical background connecting to the textbook chapters
- Step-by-step code implementations
- Visualizations and interpretations
- Exercises for further exploration

The tutorials are designed to be worked through sequentially within each notebook, but can be adapted for classroom use or self-study.

## Citation

If you find these tutorials useful in your research or teaching, please cite the accompanying textbook:

```bibtex
@article{ting2025statistical,
  title={Statistical Machine Learning for Astronomy},
  author={Ting, Yuan-Sen},
  journal={arXiv preprint arXiv:2506.12230},
  year={2025}
}
```

## License

© 2025 Yuan-Sen Ting. All rights reserved.

These tutorials may be redistributed by sharing the original GitHub repository link for educational purposes. Any other reproduction or adaptation requires explicit permission from the author.
