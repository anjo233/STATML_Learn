# Statistical Machine Learning for Astronomy: Tutorial Repository

This repository contains companion tutorials for the textbook **[Statistical Machine Learning for Astronomy](https://arxiv.org/abs/2506.12230)** by Yuan-Sen Ting.

## Overview

These tutorials provide hands-on, practical implementations of the concepts covered in the textbook. Each tutorial is designed as a standalone Jupyter notebook that explores key statistical and machine learning concepts through real astronomical applications.

## Tutorial Status

We are currently in the process of cleaning up and standardizing all tutorials. The tutorials listed below have been fully revised and are ready for use. Additional tutorials will be added as they are completed.

## Available Tutorials

### Chapter 2a: Probabilistic Distribution
- **File**: `tutorial_chapter2a.ipynb`
- **Topics**: Probability distributions, Poisson processes, maximum likelihood estimation, spatial statistics
- **Application**: Detecting stellar clusters through statistical analysis of spatial point patterns

### Chapter 2b: Bayesian Inference
- **File**: `tutorial_chapter2b.ipynb`
- **Topics**: Bayesian inference, likelihood functions, posterior distributions, marginalization over nuisance parameters, grid-based inference, uncertainty propagation
- **Application**: Inferring binary star eccentricities from single-epoch velocity-position angle measurements
  

### Chapter 3: Statistical Foundations and Summary Statistics
- **File**: `tutorial_chapter3.ipynb`
- **Topics**: Statistical moments, correlation functions, bootstrap methods, two-point statistics
- **Application**: Detecting the Baryon Acoustic Oscillation signal in simulated cosmological data

### Chapter 4: Linear Regression
- **File**: `tutorial_chapter4.ipynb`
- **Topics**: Maximum likelihood estimation for linear regression, regularization (L2/Ridge regression), feature engineering with basis functions, model evaluation with train/test splits
- **Application**: Predicting stellar properties from APOGEE infrared spectra

### Chapter 7: Classification and Logistic Regression
- **File**: `tutorial_chapter7.ipynb`
- **Topics**: Logistic regression, sigmoid transformation, gradient descent optimization, classification metrics, hyperparameter tuning
- **Application**: Distinguishing Red Clump from Red Giant Branch stars using stellar parameters from APOGEE

### Chapter 9: Bayesian Logistic Regression
- **File**: `tutorial_chapter9.ipynb`
- **Topics**: Bayesian inference for classification, Laplace approximation, predictive distributions, uncertainty quantification, prior specification
- **Application**: Quantifying classification uncertainty for Red Clump vs Red Giant Branch stars with parameter uncertainty propagation

### Chapter 10: Principal Component Analysis
- **File**: `tutorial_chapter10.ipynb`
- **Topics**: Principal Component Analysis, Singular Value Decomposition, eigendecomposition, dimensionality reduction, variance analysis, image reconstruction
- **Application**: Analyzing galaxy morphology from Hyper Suprime-Cam images to identify fundamental patterns of variation and achieve efficient data compression

### Chapter 11: K-means and Gaussian Mixture Models
- **File**: `tutorial_chapter11a.ipynb`
- **Topics**: K-means clustering, K-means++ initialization, Gaussian Mixture Models, Expectation-Maximization algorithm, model selection with AIC/BIC
- **Application**: Identifying stellar populations in the Sculptor dwarf galaxy through chemical abundance clustering to reveal episodic star formation history

### Chapters 10-11: Dimensionality Reduction and Mixture Models for Quasar Spectral Analysis
- **File**: `tutorial_chapter11b.ipynb`
- **Topics**: Principal Component Analysis for extreme dimensionality reduction, physical interpretation of principal components, Gaussian Mixture Models in reduced parameter spaces, generative modeling for synthetic spectra, likelihood-based outlier detection
- **Application**: Analyzing quasar spectra from simulated datasets and identifying unusual objects through probabilistic modeling
  

### Chapter 12: Sampling and Monte Carlo Methods
- **File**: `tutorial_chapter12.ipynb`
- **Topics**: Inverse transform sampling, rejection sampling, importance sampling, Monte Carlo integration, effective sample size
- **Application**: Generating realistic stellar populations using the Kroupa Initial Mass Function and estimating population properties

### Chapter 13: Markov Chain Monte Carlo
- **File**: `tutorial_chapter13.ipynb`
- **Topics**: Metropolis-Hastings algorithm, Gibbs sampling, convergence diagnostics (Geweke and Gelman-Rubin tests), autocorrelation analysis, effective sample size, proposal tuning, burn-in and thinning


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
