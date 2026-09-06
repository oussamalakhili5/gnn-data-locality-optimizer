# 📘 Methodology

## Overview

This document describes the methodology followed to build and evaluate the GNN-based data locality optimization model. The approach is divided into five main phases:

1. Synthetic data generation
2. Big Data integration with HDFS and Spark
3. Heterogeneous graph construction
4. GNN training and validation
5. Evaluation and visualization

## 1. Synthetic Data Generation

### Objectives
- Simulate a realistic Hadoop/Spark cluster.
- Produce a dataset that contains learnable patterns for the GNN.
- Avoid dependency on a real production cluster during development.

### Data distributions
- **DataNode capacity**: log-normal distribution.
- **Block popularity**: Zipf distribution (80/20 Pareto principle).
- **CPU/memory load**: Beta distribution.
- **Network latency**: exponential distribution.
- **Access frequency**: Poisson distribution.

### Label creation
A placement is considered **local (1)** if the compatibility score between a block and a DataNode exceeds a given threshold. The compatibility score is computed from:
- block popularity
- DataNode available resources (CPU, memory, capacity, bandwidth)

This ensures that the labels are not random and that the GNN can learn a meaningful pattern.

## 2. Big Data Integration (HDFS & Spark)

### Storage
The generated CSV files are stored in HDFS under `/data/raw`.

### Processing
PySpark is used to:
- read raw CSV files from HDFS or local disk
- aggregate access patterns
- compute rack-level load statistics
- prepare structured data for graph construction

### Execution modes
- **Local mode**: `python src/spark_processor.py`
- **HDFS mode**: `python src/spark_processor.py --hdfs`

## 3. Heterogeneous Graph Construction

The processed data is transformed into a heterogeneous graph with PyTorch Geometric.

### Node types
| Node type | Features |
|-----------|----------|
| `datanode` | capacity, CPU usage, memory usage, bandwidth, latency |
| `block` | size, popularity, access frequency, replication factor |
| `job` | priority, duration, cores, memory, shuffle read/write |

### Edge types
| Edge type | Description |
|-----------|-------------|
| `(block, placed_on, datanode)` | Physical placement of a block on a DataNode |
| `(job, accesses, block)` | Job access to a data block |

### Edge labels
The `placed_on` edges carry a binary label:
- `1` : local placement (good)
- `0` : remote placement (bad)

## 4. GNN Training and Validation

### Model architecture
- **Type**: Heterogeneous GraphSAGE
- **Layers**: 3 heterogeneous convolution layers
- **Hidden channels**: 64
- **Normalization**: BatchNorm per node type
- **Dropout**: 0.3

### Training strategy
- **Split**: 70% train, 15% validation, 15% test (stratified)
- **Class imbalance handling**: inverse-frequency class weights in CrossEntropyLoss
- **Optimizer**: Adam (learning rate = 0.001, weight decay = 5e-4)
- **Early stopping**: patience 20 epochs, monitored on validation F1
- **Checkpointing**: best model saved to `data/models/best_model.pt`

### Metrics tracked
- Loss
- Accuracy
- Precision
- Recall
- F1-score

## 5. Evaluation and Visualization

The final evaluation is performed on the held-out test set.

### Metrics computed
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve and AUC

### Generated figures
- `results/figures/confusion_matrix.png`
- `results/figures/learning_curves.png`
- `results/figures/roc_curve.png`

## 6. Results Summary

| Metric | Value |
|--------|-------|
| Accuracy | 97.77% |
| Precision | 93.55% |
| Recall | 98.31% |
| F1-score | 95.87% |

## 7. Limitations and Future Work

### Limitations
- The model is validated on synthetic data.
- The model classifies existing placements rather than generating a full optimal layout.
- Real-time constraints have not been tested.

### Future work
- Validate on a real Hadoop/Spark cluster.
- Experiment with other GNN architectures (GAT, HGT).
- Integrate the model with a data placement orchestrator (YARN/Kubernetes).
- Add temporal features to capture evolving access patterns.