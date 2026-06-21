# 💤 Temporal Characterization and Deep Learning-Based Modeling of Human Sleep

## Overview

This project explores human sleep as a temporally evolving physiological process using polysomnography (PSG) data. By combining clustering, temporal dependency modeling, and deep learning, the study investigates sleep behavior patterns, physiological signal interactions, and sleep-stage prediction.

## Dataset

The study uses the Sleep-EDF Expanded Dataset containing overnight sleep recordings with:

- EEG, EOG, and EMG signals
- Expert-labeled sleep stages
- 78 subjects across a wide age range

## Methodology

**1. Population Segmentation & Clustering**
Identified distinct sleep behavior profiles across individuals using subject-level characteristics.

**2. Temporal Dependency Modeling**
Applied Vector AutoRegression (VAR) to examine temporal relationships between EEG, EOG, and EMG signals.

**3. Sequence-Based Sleep Staging**
Developed a Bidirectional LSTM (BiLSTM) model to predict sleep stages using temporal context and compared its performance with an XGBoost baseline.

## Results

- Identified five distinct sleep behavior profiles.
- Revealed meaningful temporal interactions among physiological signals.
- Improved sleep-stage classification performance using temporal sequence modeling.

| Model | Accuracy | F1 Score |
|---------|----------|----------|
| XGBoost | 74% | 0.57 |
| BiLSTM | 84% | 0.65 |


## Applications

- Sleep health monitoring
- Sleep disorder research
- Healthcare analytics
- AI-assisted sleep analysis

## Dataset

[Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/)
