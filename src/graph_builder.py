"""
============================================================================
MODULE : Graph Builder for Heterogeneous GNN
============================================================================
Author : Oussama Lakhili
Date : 2026-08-28
Description :
    This module transforms tabular data (CSV files) into a heterogeneous
    graph structure using PyTorch Geometric. The graph represents the
    distributed system with three node types (DataNode, Block, Job) and
    two edge types (placed_on, accesses).
============================================================================
"""

# ============================================
# IMPORTS
# ============================================

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# GRAPH BUILDER CLASS
# ============================================

class GraphBuilder:
    """
    Builds a heterogeneous graph from distributed system data.
    
    The graph contains:
    - Nodes: DataNodes (servers), Blocks (data units), Jobs (tasks)
    - Edges: Block → placed_on → DataNode, Job → accesses → Block
    
    Features are normalized and labels are created for edge classification.
    """
    
    def __init__(
        self,
        data_dir: str = 'data/raw',
        locality_threshold: float = 0.7
    ):
        """
        Initialize the GraphBuilder.
        
        Args:
            data_dir (str): Directory containing the CSV files.
            locality_threshold (float): Threshold to determine good placement.
        """
        self.data_dir = data_dir
        self.locality_threshold = locality_threshold
        self.scaler = MinMaxScaler()
        
        # Store node features
        self.datanode_features = None
        self.block_features = None
        self.job_features = None
        
    def load_data(self) -> Tuple[pd.DataFrame, ...]:
        """
        Load all CSV files from the data directory.
        
        Returns:
            Tuple of DataFrames: (datanodes, blocks, jobs, access_patterns, placements)
        """
        logger.info("Loading data from CSV files...")
        
        datanodes = pd.read_csv(os.path.join(self.data_dir, 'datanodes.csv'))
        blocks = pd.read_csv(os.path.join(self.data_dir, 'blocks.csv'))
        jobs = pd.read_csv(os.path.join(self.data_dir, 'jobs.csv'))
        access_patterns = pd.read_csv(os.path.join(self.data_dir, 'access_patterns.csv'))
        placements = pd.read_csv(os.path.join(self.data_dir, 'placements.csv'))
        
        logger.info(f"Data loaded: {len(datanodes)} DataNodes, {len(blocks)} Blocks, {len(jobs)} Jobs")
        
        return datanodes, blocks, jobs, access_patterns, placements
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to range [0, 1] using MinMaxScaler.
        
        Args:
            features (np.ndarray): Raw features to normalize.
            
        Returns:
            np.ndarray: Normalized features.
        """
        return self.scaler.fit_transform(features)
    
    def extract_datanode_features(self, datanodes: pd.DataFrame) -> torch.Tensor:
        """
        Extract and normalize features for DataNode nodes.
        
        Features: [capacity, cpu_usage, memory_usage, bandwidth, latency]
        
        Args:
            datanodes (pd.DataFrame): DataNode data.
            
        Returns:
            torch.Tensor: Normalized features tensor.
        """
        # Select relevant columns
        feature_columns = [
            'capacity_tb',
            'cpu_usage',
            'memory_usage',
            'bandwidth_gbps',
            'network_latency_ms'
        ]
        
        # Extract features
        features = datanodes[feature_columns].values
        
        # Normalize
        features_normalized = self.normalize_features(features)
        
        logger.info(f"DataNode features: {features_normalized.shape}")
        
        return torch.tensor(features_normalized, dtype=torch.float)
    
    def extract_block_features(self, blocks: pd.DataFrame) -> torch.Tensor:
        """
        Extract and normalize features for Block nodes.
        
        Features: [size, popularity, access_frequency, replication_factor]
        
        Args:
            blocks (pd.DataFrame): Block data.
            
        Returns:
            torch.Tensor: Normalized features tensor.
        """
        # Select relevant columns
        feature_columns = [
            'size_mb',
            'popularity_score',
            'access_frequency',
            'replication_factor'
        ]
        
        # Extract features
        features = blocks[feature_columns].values
        
        # Normalize
        features_normalized = self.normalize_features(features)
        
        logger.info(f"Block features: {features_normalized.shape}")
        
        return torch.tensor(features_normalized, dtype=torch.float)
    
    def extract_job_features(self, jobs: pd.DataFrame) -> torch.Tensor:
        """
        Extract and normalize features for Job nodes.
        
        Features: [priority, duration, cores, memory, shuffle_read, shuffle_write]
        
        Args:
            jobs (pd.DataFrame): Job data.
            
        Returns:
            torch.Tensor: Normalized features tensor.
        """
        # Encode priority as numeric values
        priority_map = {'LOW': 0.0, 'MEDIUM': 0.5, 'HIGH': 1.0}
        jobs['priority_encoded'] = jobs['priority'].map(priority_map)
        
        # Select relevant columns
        feature_columns = [
            'priority_encoded',
            'duration_minutes',
            'executor_cores',
            'executor_memory_gb',
            'shuffle_read_mb',
            'shuffle_write_mb'
        ]
        
        # Extract features
        features = jobs[feature_columns].values
        
        # Normalize
        features_normalized = self.normalize_features(features)
        
        logger.info(f"Job features: {features_normalized.shape}")
        
        return torch.tensor(features_normalized, dtype=torch.float)
    
    def create_placed_on_edges(
        self,
        placements: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edges between Blocks and DataNodes (placed_on relationship).
        
        Args:
            placements (pd.DataFrame): Placement data.
            
        Returns:
            Tuple of (edge_index, edge_labels)
            - edge_index: [2, num_edges] tensor of (block_id, datanode_id)
            - edge_labels: [num_edges] tensor of binary labels (1=good, 0=bad)
        """
        # Extract edge indices
        # edge_index = torch.tensor(
        #     [placements['block_id'].values, placements['datanode_id'].values],
        #     dtype=torch.long
        # )
        edge_index = torch.tensor(
            np.array([placements['block_id'].values, placements['datanode_id'].values]),
            dtype=torch.long
       )
        
        # Create labels (1 = local/good placement, 0 = remote/bad placement)
        edge_labels = torch.tensor(
            placements['is_local'].values.astype(int),
            dtype=torch.long
        )
        
        logger.info(f"Placed_on edges: {edge_index.shape[1]}")
        logger.info(f"Edge labels: {edge_labels.sum().item()} positive, {(edge_labels == 0).sum().item()} negative")
        
        return edge_index, edge_labels
    
    def create_access_edges(
        self,
        access_patterns: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edges between Jobs and Blocks (accesses relationship).
        
        Args:
            access_patterns (pd.DataFrame): Access pattern data.
            
        Returns:
            Tuple of (edge_index, edge_weights)
            - edge_index: [2, num_edges] tensor of (job_id, block_id)
            - edge_weights: [num_edges] tensor of access frequencies
        """
        # Extract edge indices
        # edge_index = torch.tensor(
        #     [access_patterns['job_id'].values, access_patterns['block_id'].values],
        #     dtype=torch.long
        # )
        edge_index = torch.tensor(
            np.array([access_patterns['job_id'].values, access_patterns['block_id'].values]),
            dtype=torch.long
        )
        
        # Edge weights based on access frequency
        edge_weights = torch.tensor(
            access_patterns['access_count'].values,
            dtype=torch.float
        )
        
        logger.info(f"Access edges: {edge_index.shape[1]}")
        
        return edge_index, edge_weights
    
    def build_heterogeneous_graph(self) -> HeteroData:
        """
        Build the complete heterogeneous graph.
        
        Returns:
            HeteroData: PyTorch Geometric heterogeneous graph.
        """
        logger.info("Building heterogeneous graph...")
        
        # Load data
        datanodes, blocks, jobs, access_patterns, placements = self.load_data()
        
        # Extract node features
        self.datanode_features = self.extract_datanode_features(datanodes)
        self.block_features = self.extract_block_features(blocks)
        self.job_features = self.extract_job_features(jobs)
        
        # Create edges
        placed_on_edges, placed_on_labels = self.create_placed_on_edges(placements)
        access_edges, access_weights = self.create_access_edges(access_patterns)
        
        # Build the heterogeneous graph
        graph = HeteroData()
        
        # Add node features
        graph['datanode'].x = self.datanode_features
        graph['block'].x = self.block_features
        graph['job'].x = self.job_features
        
        # Add edges
        graph['block', 'placed_on', 'datanode'].edge_index = placed_on_edges
        graph['block', 'placed_on', 'datanode'].y = placed_on_labels
        
        graph['job', 'accesses', 'block'].edge_index = access_edges
        graph['job', 'accesses', 'block'].edge_weight = access_weights
        
        # Store metadata
        graph.graph_info = {
            'num_datanodes': self.datanode_features.shape[0],
            'num_blocks': self.block_features.shape[0],
            'num_jobs': self.job_features.shape[0],
            'num_placed_on_edges': placed_on_edges.shape[1],
            'num_access_edges': access_edges.shape[1],
            'locality_threshold': self.locality_threshold
        }
        
        logger.info(f"  - DataNodes: {graph.graph_info['num_datanodes']}")
        logger.info(f"  - Blocks: {graph.graph_info['num_blocks']}")
        logger.info(f"  - Jobs: {graph.graph_info['num_jobs']}")
        logger.info(f"  - Placed_on edges: {graph.graph_info['num_placed_on_edges']}")
        logger.info(f"  - Access edges: {graph.graph_info['num_access_edges']}")
            
        return graph
    
    def save_graph(self, graph: HeteroData, output_path: str = 'data/graphs/graph.pt') -> None:
        """
        Save the graph to disk.
        
        Args:
            graph (HeteroData): The graph to save.
            output_path (str): Path to save the graph.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(graph, output_path)
        logger.info(f"Graph saved to {output_path}")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Create the graph builder
    builder = GraphBuilder(data_dir='data/raw', locality_threshold=0.7)
    
    # Build the graph
    graph = builder.build_heterogeneous_graph()
    
    # Save the graph
    builder.save_graph(graph)
    
    # Display graph summary
    print("\n" + "=" * 50)
    print("GRAPH SUMMARY")
    print("=" * 50)
    print(f"Node types: {graph.node_types}")
    print(f"Edge types: {graph.edge_types}")
    print(f"DataNode features shape: {graph['datanode'].x.shape}")
    print(f"Block features shape: {graph['block'].x.shape}")
    print(f"Job features shape: {graph['job'].x.shape}")
    print(f"Placed_on edges: {graph['block', 'placed_on', 'datanode'].edge_index.shape}")
    print(f"Access edges: {graph['job', 'accesses', 'block'].edge_index.shape}")
    print("=" * 50)
    