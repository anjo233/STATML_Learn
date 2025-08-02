# Statistical Machine Learning for Astronomy: Tutorial Repository

This repository contains companion tutorials for the textbook **[Statistical Machine Learning for Astronomy](https://arxiv.org/abs/2506.12230)** by Yuan-Sen Ting.

## Overview

These tutorials provide hands-on, practical implementations of the concepts covered in the textbook. Each tutorial is designed as a standalone Jupyter notebook that explores key statistical and machine learning concepts through real astronomical applications.

## Tutorial Status

We are currently in the process of cleaning up and standardizing all tutorials. The tutorials listed below have been fully revised and are ready for use. Additional tutorials will be added as they are completed.

## Available Tutorials

### Chapter 2a: Probabilistic Distribution
- **File**: `tutorial_chapter_2a.ipynb`
- **Topics**: Probability distributions, Poisson processes, maximum likelihood estimation, spatial statistics
- **Application**: Detecting stellar clusters through statistical analysis of spatial point patterns

### Chapter 2b: Bayesian Inference
- **File**: `tutorial_chapter_2b.ipynb`
- **Topics**: Bayesian inference, likelihood functions, posterior distributions, marginalization over nuisance parameters, grid-based inference, uncertainty propagation
- **Application**: Inferring binary star eccentricities from single-epoch velocity-position angle measurements
  

### Chapter 3: Statistical Foundations and Summary Statistics
- **File**: `tutorial_chapter_3.ipynb`
- **Topics**: Statistical moments, correlation functions, bootstrap methods, two-point statistics
- **Application**: Detecting the Baryon Acoustic Oscillation signal in simulated cosmological data

### Chapter 4a: Linear Regression
- **File**: `tutorial_chapter_4a.ipynb`
- **Topics**: Maximum likelihood estimation for linear regression, regularization (L2/Ridge regression), feature engineering with basis functions, model evaluation with train/test splits
- **Application**: Predicting stellar properties from APOGEE infrared spectra

### Chapter 4b: Linear Regression
- **File**: `tutorial_chapter_4b.ipynb`
- **Topics**: Calibration as regression, weighted least squares with heteroscedastic uncertainties, bootstrap uncertainty analysis, sparse design matrices
- **Application**: Calibrating radial velocity measurements from telescope networks using standard stars to correct for systematic instrumental and atmospheric effects

### Chapter 5: Bayesian Linear Regression
- **File**: `tutorial_chapter_5.ipynb`
- **Topics**: Bayesian linear regression, heteroscedastic noise modeling, conjugate priors, posterior distributions, predictive uncertainty quantification, model calibration, uncertainty decomposition
- **Application**: Predicting stellar temperatures from APOGEE spectra with properly calibrated uncertainties, demonstrating how Bayesian methods provide principled uncertainty quantification beyond point estimates

### Chapter 7: Classification and Logistic Regression
- **File**: `tutorial_chapter_7.ipynb`
- **Topics**: Logistic regression, sigmoid transformation, gradient descent optimization, classification metrics, hyperparameter tuning
- **Application**: Distinguishing Red Clump from Red Giant Branch stars using stellar parameters from APOGEE

### Chapter 8: Multi-class Classification
- **File**: `tutorial_chapter_8.ipynb`
- **Topics**: Softmax regression, multi-class logistic regression, cross-entropy loss, feature extraction from images, confusion matrices, classification metrics
- **Application**: Classifying strong gravitational lensing features in James Webb Space Telescope images using pre-trained vision models and multi-class logistic regression

### Chapter 9: Bayesian Logistic Regression
- **File**: `tutorial_chapter_9.ipynb`
- **Topics**: Bayesian inference for classification, Laplace approximation, predictive distributions, uncertainty quantification, prior specification
- **Application**: Quantifying classification uncertainty for Red Clump vs Red Giant Branch stars with parameter uncertainty propagation

### Chapter 10: Principal Component Analysis
- **File**: `tutorial_chapter_10.ipynb`
- **Topics**: Principal Component Analysis, Singular Value Decomposition, eigendecomposition, dimensionality reduction, variance analysis, image reconstruction
- **Application**: Analyzing galaxy morphology from Hyper Suprime-Cam images to identify fundamental patterns of variation and achieve efficient data compression

### Chapter 11: K-means and Gaussian Mixture Models
- **File**: `tutorial_chapter_11a.ipynb`
- **Topics**: K-means clustering, K-means++ initialization, Gaussian Mixture Models, Expectation-Maximization algorithm, model selection with AIC/BIC
- **Application**: Identifying stellar populations in the Sculptor dwarf galaxy through chemical abundance clustering to reveal episodic star formation history

### Chapters 10-11: Dimensionality Reduction and Mixture Models for Quasar Spectral Analysis
- **File**: `tutorial_chapter_11b.ipynb`
- **Topics**: Principal Component Analysis for extreme dimensionality reduction, physical interpretation of principal components, Gaussian Mixture Models in reduced parameter spaces, generative modeling for synthetic spectra, likelihood-based outlier detection
- **Application**: Analyzing quasar spectra from simulated datasets and identifying unusual objects through probabilistic modeling
  

### Chapter 12: Sampling and Monte Carlo Methods
- **File**: `tutorial_chapter_12.ipynb`
- **Topics**: Inverse transform sampling, rejection sampling, importance sampling, Monte Carlo integration, effective sample size
- **Application**: Generating realistic stellar populations using the Kroupa Initial Mass Function and estimating population properties

### Chapter 13: Markov Chain Monte Carlo
- **File**: `tutorial_chapter_13.ipynb`
- **Topics**: Metropolis-Hastings algorithm, Gibbs sampling, convergence diagnostics (Geweke and Gelman-Rubin tests), autocorrelation analysis, effective sample size, proposal tuning, burn-in and thinning

### Chapter 14a: Gaussian Process Regression
- **File**: `tutorial_chapter_14a.ipynb`
- **Topics**: Gaussian Process regression, kernel functions, marginal likelihood optimization, hyperparameter learning, Cholesky decomposition, predictive uncertainty quantification
- **Application**: Analyzing Kepler stellar light curves to extract oscillation timescales through GP regression, revealing the relationship between stellar surface gravity and asteroseismic properties
  
### Chapter 14b: Gaussian Process Classification
- **File**: `tutorial_chapter_14b.ipynb`
- **Topics**: Gaussian Process Classification, latent variable models, Laplace approximation, fixed-point iteration, kernel hyperparameter selection
- **Application**: Star-galaxy separation using Gaia photometric colors, demonstrating how GP classification learns flexible nonlinear decision boundaries

### Chapter 15a: Backpropagation and Introduction to PyTorch
- **File**: `tutorial_chapter_15a.ipynb`
- **Topics**: Backpropagation implementation from scratch, automatic differentiation, computational graphs, PyTorch tensors and autograd, optimizer comparison (SGD vs Adam), gradient flow visualization

### Chapter 15b: Autoencoders
- **File**: `tutorial_chapter_15b.ipynb`
- **Topics**: Autoencoder architecture, encoder-decoder networks, latent representations, latent space visualization, interpolation in latent space, anomaly detection through reconstruction error
- **Application**: Analyzing galaxy morphology from Hyper Suprime-Cam images through nonlinear dimension reduction, demonstrating how autoencoders extend beyond PCA's linear constraints to capture complex morphological relationships

### Chapter 15c: Mixture Density Networks
- **File**: `tutorial_chapter_15c.ipynb`
- **Topics**: Mixture Density Networks, conditional density estimation, probabilistic regression with neural networks, Gaussian mixture outputs, maximum likelihood training, multimodal predictions, uncertainty quantification
- **Application**: Modeling stellar lithium abundance in open clusters as a function of effective temperature and age, demonstrating how MDNs capture both systematic trends and intrinsic astrophysical scatter that deterministic models miss

### Chapter 15d: Normalizing Flows
- **File**: `tutorial_chapter_15d.ipynb`
- **Topics**: Normalizing flows, RealNVP architecture, invertible neural networks, change of variables formula, Jacobian computation, coupling layers, likelihood-free inference, generative modeling
- **Application**: Modeling the 13-dimensional chemical abundance distribution of stars from APOGEE survey data, demonstrating how normalizing flows learn complex multimodal distributions without parametric assumptions and enable both density estimation and sample generation

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

The tutorials are designed to be worked through sequentially within each notebook.

## Citation

If you find these resources useful in your research or teaching, please cite the textbook or this tutorial repository.

### Textbook

```bibtex
@article{ting2025statistical,
  title={Statistical Machine Learning for Astronomy},
  author={Ting, Yuan-Sen},
  journal={arXiv preprint arXiv:2506.12230},
  year={2025}
}
```

### Tutorials

```bibtex
@software{ting2025statisticaltutorial,
  author       = {Ting, Yuan-Sen},
  title        = {{tingyuansen/statml: Statistical Machine Learning 
                 for Astronomy - Tutorials (v1.0)}},
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.16495692},
  url          = {https://doi.org/10.5281/zenodo.16495692}
}
```

## License

© 2025 Yuan-Sen Ting. All rights reserved.

These tutorials may be redistributed by sharing the original GitHub repository link for educational purposes. Any other reproduction or adaptation requires explicit permission from the author.



---

# 面向天文学的统计机器学习：教程仓库

本仓库包含《[面向天文学的统计机器学习](https://arxiv.org/abs/2506.12230)》（作者：Yuan-Sen Ting）教材的配套教程。

## 概述

这些教程为教材中的概念提供了实际的、可操作的实现。每个教程都是独立的 Jupyter Notebook，深入探索教材中的关键统计和机器学习方法，并结合实际的天文学应用案例。

## 教程进展

目前我们正在对所有教程进行清理和标准化。下表中的教程已完整修订并可直接使用，更多教程正在陆续添加中。

## 已发布教程

### 第2a章：概率分布
- **文件**：`tutorial_chapter_2a.ipynb`
- **主题**：概率分布、泊松过程、最大似然估计、空间统计
- **应用**：通过空间点模式统计分析检测恒星团

### 第2b章：贝叶斯推断
- **文件**：`tutorial_chapter_2b.ipynb`
- **主题**：贝叶斯推断、似然函数、后验分布、对干扰参数的边际化、基于网格的推断、不确定性传播
- **应用**：从单次速度-位置角度测量推断双星偏心率

### 第3章：统计基础与总结统计量
- **文件**：`tutorial_chapter_3.ipynb`
- **主题**：统计矩、相关函数、自助法、两点统计
- **应用**：在模拟宇宙学数据中检测重子声学振荡信号

### 第4a章：线性回归
- **文件**：`tutorial_chapter_4a.ipynb`
- **主题**：线性回归的最大似然估计、正则化（L2/岭回归）、特征工程与基函数、模型评估（训练/测试集划分）
- **应用**：利用 APOGEE 红外光谱预测恒星属性

### 第4b章：线性回归
- **文件**：`tutorial_chapter_4b.ipynb`
- **主题**：回归中的校准、具有异方差不确定性的加权最小二乘、自助法不确定性分析、稀疏设计矩阵
- **应用**：利用标准恒星校准望远镜网络的径向速度测量，以校正仪器和大气的系统误差

### 第5章：贝叶斯线性回归
- **文件**：`tutorial_chapter_5.ipynb`
- **主题**：贝叶斯线性回归、异方差噪声建模、共轭先验、后验分布、预测不确定性量化、模型校准、不确定性分解
- **应用**：基于 APOGEE 光谱预测恒星温度，并通过贝叶斯方法实现更可靠的不确定性量化

### 第7章：分类与逻辑回归
- **文件**：`tutorial_chapter_7.ipynb`
- **主题**：逻辑回归、S型变换、梯度下降优化、分类指标、超参数调优
- **应用**：利用 APOGEE 恒星参数区分红团星与红巨星分支星

### 第8章：多类分类
- **文件**：`tutorial_chapter_8.ipynb`
- **主题**：Softmax 回归、多类逻辑回归、交叉熵损失、图像特征提取、混淆矩阵、分类指标
- **应用**：利用预训练视觉模型与多类逻辑回归对詹姆斯·韦伯空间望远镜图像中的强引力透镜特征进行分类

### 第9章：贝叶斯逻辑回归
- **文件**：`tutorial_chapter_9.ipynb`
- **主题**：分类的贝叶斯推断、拉普拉斯近似、预测分布、不确定性量化、先验设定
- **应用**：对红团星与红巨星分支星的分类不确定性进行量化

### 第10章：主成分分析
- **文件**：`tutorial_chapter_10.ipynb`
- **主题**：主成分分析、奇异值分解、特征分解、降维、方差分析、图像重构
- **应用**：通过 Hyper Suprime-Cam 图像分析星系形态，识别主要变化模式并实现高效数据压缩

### 第11章：K-均值与高斯混合模型
- **文件**：`tutorial_chapter_11a.ipynb`
- **主题**：K-均值聚类、K-means++ 初始化、高斯混合模型、期望最大化算法、AIC/BIC 模型选择
- **应用**：通过化学丰度聚类识别 Sculptor 矮星系中的恒星族群，揭示星系的阶段性形成历史

### 第10-11章：降维与混合模型在类星体光谱分析中的应用
- **文件**：`tutorial_chapter_11b.ipynb`
- **主题**：极端降维的主成分分析、主成分的物理解释、低维参数空间中的高斯混合模型、生成建模
- **应用**：分析模拟数据集中的类星体光谱，通过概率建模识别异常天体

### 第12章：采样与蒙特卡洛方法
- **文件**：`tutorial_chapter_12.ipynb`
- **主题**：逆变换采样、拒绝采样、重要性采样、蒙特卡洛积分、有效样本量
- **应用**：利用 Kroupa 初始质量函数生成真实的恒星族群并估算族群属性

### 第13章：马尔可夫链蒙特卡洛
- **文件**：`tutorial_chapter_13.ipynb`
- **主题**：Metropolis-Hastings 算法、Gibbs 采样、收敛诊断（Geweke、Gelman-Rubin）、自相关分析、有效样本量、建议分布调优、预热与抽样间隔

### 第14a章：高斯过程回归
- **文件**：`tutorial_chapter_14a.ipynb`
- **主题**：高斯过程回归、核函数、边际似然优化、超参数学习、Cholesky 分解、预测不确定性量化
- **应用**：利用 GP 回归分析 Kepler 恒星光变曲线，提取振荡时间尺度，揭示恒星表面重力与星震属性的关系

### 第14b章：高斯过程分类
- **文件**：`tutorial_chapter_14b.ipynb`
- **主题**：高斯过程分类、潜变量模型、拉普拉斯近似、定点迭代、核超参数选择
- **应用**：利用 Gaia 光度颜色实现星-星系分离，展示 GP 分类灵活学习非线性决策边界的能力

### 第15a章：反向传播与 PyTorch 入门
- **文件**：`tutorial_chapter_15a.ipynb`
- **主题**：从零实现反向传播、自动微分、计算图、PyTorch 张量与自动求导、优化器比较（SGD 与 Adam）、梯度流可视化

### 第15b章：自动编码器
- **文件**：`tutorial_chapter_15b.ipynb`
- **主题**：自动编码器结构、编码器-解码器网络、潜在表示、潜空间可视化、潜空间插值、通过重构误差进行异常检测
- **应用**：通过自动编码器对 Hyper Suprime-Cam 图像中的星系形态进行非线性降维，突破 PCA 的线性约束，捕获更复杂的形态特征

### 第15c章：混合密度网络
- **文件**：`tutorial_chapter_15c.ipynb`
- **主题**：混合密度网络、条件密度估计、基于神经网络的概率回归、高斯混合输出、最大似然训练、多峰预测、不确定性量化
- **应用**：建模开放星团的恒星锂丰度与有效温度、年龄的关系，展示 MDN 如何同时捕捉系统性趋势和内在天体物理变化

### 第15d章：归一化流
- **文件**：`tutorial_chapter_15d.ipynb`
- **主题**：归一化流、RealNVP 结构、可逆神经网络、变量变换公式、雅可比计算、耦合层、无似然推断、生成建模
- **应用**：建模 APOGEE 调查数据中13维化学丰度分布，展示归一化流如何无参数地学习复杂多峰分布

## 运行环境

要运行这些教程，你需要：
- Python 3.7 或更高版本
- NumPy
- Matplotlib
- SciPy
- Jupyter Notebook 或 JupyterLab

## 使用方法

每个教程均为自包含，内容包括：
- 与教材章节相关的理论背景
- 逐步代码实现
- 可视化与结果解读

建议按顺序在每个 Notebook 内依次学习。

## 引用

如果你在科研或教学中觉得这些资源有用，请引用教材或本教程仓库。

### 教材

```bibtex
@article{ting2025statistical,
  title={Statistical Machine Learning for Astronomy},
  author={Ting, Yuan-Sen},
  journal={arXiv preprint arXiv:2506.12230},
  year={2025}
}
```

### 教程

```bibtex
@software{ting2025statisticaltutorial,
  author       = {Ting, Yuan-Sen},
  title        = {{tingyuansen/statml: Statistical Machine Learning 
                 for Astronomy - Tutorials (v1.0)}},
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.16495692},
  url          = {https://doi.org/10.5281/zenodo.16495692}
}
```

## 版权声明

© 2025 Yuan-Sen Ting。保留所有权利。

这些教程可通过分享原 GitHub 仓库链接用于教育用途。任何其他形式的转载或改编需获得作者的明确许可。

---
