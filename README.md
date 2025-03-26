---

# Chinese Semantic Reasoning Based on BERT

![Python](https://img.shields.io/badge/Python-3.8-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-1.13-orange.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A BERT-based deep learning project for Chinese Natural Language Inference (NLI), classifying entailment, neutral, and contradiction in text pairs using the CINLID dataset. Features training, evaluation, inference, and visualizations.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Results](#results)
7. [Visualizations](#visualizations)
8. [License](#license)

---

## Project Overview
*Chinese Semantic Reasoning Based on BERT* is a project that fine-tunes a pre-trained BERT model for Natural Language Inference (NLI) on Chinese text pairs. It uses the CINLID dataset to classify relationships (entailment, neutral, contradiction) and provides tools for training, evaluation, and result visualization.

---

## Features
- Fine-tuned BERT for Chinese NLI tasks.
- Supports training with customizable epochs and learning rates.
- Inference on custom Chinese text pairs.
- Visualizations: training loss/accuracy curves, confusion matrix, classification report heatmap.
- GPU acceleration with PyTorch.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch with CUDA (optional, for GPU support)
- Required libraries:
  ```bash
  pip install torch transformers pandas scikit-learn matplotlib seaborn