# 🏗️ Technical Architecture

## Overview

The project implements a complete pipeline from data collection to model evaluation.

```text
+----------------+     +----------------+     +----------------+
| Data Collector | --> | HDFS Storage   | --> | Spark Processing|
+----------------+     +----------------+     +----------------+
                                                     |
                                                     v
+----------------+     +----------------+     +----------------+
| GNN Model      | <-- | Graph Builder  | <-- | Features       |
+----------------+     +----------------+     +----------------+
        |
        v
+----------------+
| Optimized      |
| Placement      |
+----------------+

Main Components
1. Data Collector (src/data_collector.py)
Generates a realistic synthetic dataset simulating a Hadoop/Spark cluster.

50 DataNodes: capacity, CPU/memory load, bandwidth, latency

1,000 Blocks: size, popularity (Zipf distribution), access frequency, replication

500 Jobs: priority, duration, resources, shuffle

4,909 access relationships: Job → Block

2,981 placements: Block → DataNode with locality labels

2. HDFS (Docker)
Distributed storage of raw data using Hadoop HDFS.

NameNode and DataNode in Docker containers

Data stored in /data/raw

Web UI: http://localhost:9870

3. Spark Processor (src/spark_processor.py)
Large-scale processing with PySpark.

Reads CSV from local disk or HDFS

Aggregates access patterns

Computes rack-level load statistics

Modes: local (no --hdfs) and HDFS (--hdfs)

4. Graph Builder (src/graph_builder.py)
Transforms tabular data into a heterogeneous PyTorch Geometric graph.

Node types:

datanode: 5 normalized features

block: 4 normalized features

job: 6 normalized features

Edge types:

('block', 'placed_on', 'datanode'): 2,981 edges with binary labels

('job', 'accesses', 'block'): 4,909 edges

5. GNN Model (src/gnn_model.py)
Heterogeneous Graph Neural Network based on GraphSAGE.

3 heterogeneous convolution layers

Batch normalization per node type

Binary edge classifier

64-dimensional embeddings

6. Trainer (src/trainer.py)
Complete training loop with:

Stratified split (70/15/15)

Class weighting to handle imbalance

Early stopping based on validation F1

Best model checkpointing

7. Evaluator (src/evaluator.py)
Final evaluation and visualizations:

Confusion matrix

Learning curves (loss, accuracy, F1)

ROC curve

Classification report

Results
Metric	Value
Accuracy	97.77%
Precision	93.55%
Recall	98.31%
F1-Score	95.87%
Technologies
Python 3.11

PyTorch 2.x

PyTorch Geometric 2.x

Apache Spark 3.3

Hadoop 3.2 (Docker)

Docker Compose

Execution
# Full pipeline in local mode
.\scripts\run_pipeline.ps1 -Mode local

# Full pipeline in HDFS mode (requires Docker)
.\scripts\run_pipeline.ps1 -Mode hdfs