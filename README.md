# Temporal Characterization and Deep Learning-Based Modeling of Human Sleep

> Exploring human sleep as a temporally evolving physiological process using polysomnography (PSG) data — combining clustering, temporal dependency modeling, and deep learning.

---

## Overview

This project investigates sleep behavior patterns, physiological signal interactions, and sleep-stage prediction through a multi-step analytical pipeline. By integrating unsupervised clustering, time-series modeling, and sequence-based deep learning, we characterize both population-level sleep profiles and individual temporal dynamics.

---

## Dataset

**Sleep-EDF Expanded Dataset** — available on [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)

| Property | Details |
|:---------|:--------|
| Subjects | 78 individuals across a wide age range |
| Signals | EEG, EOG, EMG |
| Labels | Expert-annotated sleep stages |
| Format | Overnight polysomnography (PSG) recordings |

---

## Methodology

### 1. Population Segmentation & Clustering
Identified distinct sleep behavior profiles across individuals using subject-level physiological characteristics. This step uncovers natural groupings in how people sleep.

### 2. Temporal Dependency Modeling
Applied **Vector AutoRegression (VAR)** to examine how EEG, EOG, and EMG signals influence one another over time, revealing meaningful cross-signal temporal relationships.

### 3. Sequence-Based Sleep Staging
Developed a **Bidirectional LSTM (BiLSTM)** model to classify sleep stages using sequential temporal context. Performance was benchmarked against an XGBoost baseline.

---

## Results

### Sleep Profiles
- **5 distinct sleep behavior profiles** identified across the population

### Model Performance

| Model | Accuracy | F1 Score |
|:------|:--------:|:--------:|
| XGBoost (baseline) | 74% | 0.57 |
| **BiLSTM** | **84%** | **0.65** |

### Key Findings
- **+10% improvement** in classification accuracy over baseline
- Improved detection of **transitional and minority sleep stages** through temporal sequence modeling
- Meaningful **cross-signal temporal interactions** revealed between EEG, EOG, and EMG

### ⚠️ Class Imbalance
The dataset exhibited substantial class imbalance, with **Wake (W)** as the dominant sleep stage. Class-weighted training was applied to mitigate this, though imbalance remained a challenge throughout the modeling process.

---

## Applications

- Sleep health monitoring
- Sleep disorder research
- Healthcare analytics
- AI-assisted sleep analysis
