"""
Unit tests for GraphBuilder.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from graph_builder import GraphBuilder


def test_graph_shapes():
    """Check that the graph has the expected node and edge counts."""
    builder = GraphBuilder(data_dir='data/raw')
    graph = builder.build_heterogeneous_graph()

    assert graph['datanode'].x.shape[0] == 50
    assert graph['block'].x.shape[0] == 1000
    assert graph['job'].x.shape[0] == 500

    assert ('block', 'placed_on', 'datanode') in graph.edge_types
    assert ('job', 'accesses', 'block') in graph.edge_types


def test_placed_on_labels():
    """Check that edge labels are binary (0 or 1)."""
    builder = GraphBuilder(data_dir='data/raw')
    graph = builder.build_heterogeneous_graph()

    labels = graph['block', 'placed_on', 'datanode'].y
    assert set(labels.tolist()) <= {0, 1}