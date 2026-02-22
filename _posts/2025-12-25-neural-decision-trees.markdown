---
layout: post
title: "Neural Decision Tree for Bio-TinyML"
date: 2024-12-25 10:00:00 +0000
projects: true
tags: [neural, decision-trees, tinyml, healthcare, interpretable-ml]
excerpt: "A lightweight Neural Decision Tree (NDT) architecture that combines backpropagation training with decision-tree efficiency for TinyML healthcare applications."
---

## Overview

In our work presented at the Biomedical Circuits and Systems Conference 2024, we introduce a **Neural Decision Tree (NDT)** — a hybrid architecture that combines the interpretability of decision trees with the end-to-end training capability of neural networks.

The motivation was simple:

> Can we design a model that trains like a neural network, but runs like a decision tree?

The answer is yes — and it turns out to be particularly useful for TinyML healthcare systems.

---

## The Problem

Deep Neural Networks achieve high accuracy, but they are:

- Computationally heavy  
- Memory intensive  
- Power hungry  
- Often black-box  

In TinyML healthcare deployments — especially battery-powered or rural diagnostic systems — inference cost directly translates to power consumption and device lifetime.

Decision Trees are lightweight and interpretable, but they rely on handcrafted split criteria and cannot be trained end-to-end using backpropagation.

We wanted both.

---

## Neural Decision Tree (NDT)

The Neural Decision Tree replaces classical entropy-based split rules with **learnable neural decision nodes**, trained using backpropagation.

Unlike standard neural networks:

- Only one path is activated per input (tree-style routing)
- Computation scales linearly with depth
- Parameterization remains lightweight
- The structure retains interpretability

![Neural Decision Tree Architecture]({{ '/assets/images/neural-decision-trees/ndt_architecture.png' | relative_url }})

*Neural Decision Tree architecture (depth = 2). Each node learns parameters while activating only one subtree per input.*

This routing mechanism is the key to reducing compute without sacrificing non-linearity.

---

## Does It Work?

We evaluated NDT on five biomedical datasets:

- Heart Disease  
- Parkinson’s Disease  
- Breast Cancer  
- Cervical Cancer  
- Epileptic Seizure Detection  

Across tasks, NDT performs competitively with neural networks — often within a small margin — while using significantly fewer parameters.

But the real story is compute.

---

## 10× Reduction in Compute

Multiply-accumulate operations (MACs) determine inference cost in TinyML systems.

For example, on Epileptic Seizure detection:

- Neural Network: ~11277 MACs  
- Neural Decision Tree: ~1073 MACs  

That’s over **10× reduction in compute**.

![MAC comparison NN vs NDT]({{ 'assets\images\neural-decision-trees\mac_comparison.png' | relative_url }})

*MAC comparison between Neural Networks and Neural Decision Trees across biomedical datasets.*

The efficiency gain comes directly from the tree-style routing: only one subtree is active at each depth.

---

## Why This Matters

Neural Decision Trees are:

- Backpropagation-trainable  
- Interpretable  
- Hardware-compatible  
- TinyML-friendly  

They are not meant to replace deep neural networks.

They are meant to enable machine learning where conventional neural networks are too costly.

---

## Looking Ahead

This work opens several directions:

- Ensemble NDT architectures  
- Multiclass extensions  
- Regularization strategies  
- Neuromorphic or spiking NDT variants  

We see this as a foundational step toward **efficient, interpretable edge intelligence for healthcare.**

---

## Links

- [Read the full paper](https://ieeexplore.ieee.org/abstract/document/10798396/)
- [Poster](https://drive.google.com/file/d/1iIjWx_KhRVvta1Q8llej2lbeLwNZXWKI/view?usp=drive_link)