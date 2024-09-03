# Generalization of deep learning based holographic reconstruction for histopathology

We provide pytorch(python) implementations of **Generalization of deep learning based holographic reconstruction for histopathology**. This code was written by Jiseong Barg. Last update: 2024.09.04

# Overview

Holographic microscopy has become a vital tool in the field of life sciences, offering detailed, label-free morphological imaging at the wavelength scale,
with simpler and more practical setups compared to traditional interferometric systems.
 However, conventional phase retrieval methods encounter challenges such as instability, noise sensitivity, and data redundancy. 
 While deep learning (DL)-based approaches have demonstrated superior performance, they remain constrained by the out-of-distribution (OOD) problem.
 Here, we propose a novel DL-based holographic reconstruction method integrating back-propagation and a refined forward model with effective parameterization to overcome OOD problem.
 By leveraging back-propagation and reparameterization, the proposed method enhances both shape and system adaptability, facilitating reliable phase reconstruction across diverse configurations.
 Validation with limited training data from rectum tissue demonstrated robust phase reconstruction performance across various configuration, even with the cancerous rectum tissue,
 underscoring the method's adaptability to OOD data and its potential for clinical and research applications.

 ![Overview of proposed approach](./image/main1_latest.png)
