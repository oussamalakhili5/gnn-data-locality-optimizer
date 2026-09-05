import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from graph_builder import GraphBuilder

def test_build_graph():
    builder = GraphBuilder(data_dir='data/raw')
    graph = builder.build_heterogeneous_graph()
    assert graph['datanode'].x.shape[0] == 50
    assert graph['block'].x.shape[0] == 1000
    assert graph['job'].x.shape[0] == 500