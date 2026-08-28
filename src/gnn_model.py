"""Graph neural network models for heterogeneous data locality optimization.

This module defines PyTorch Geometric models that learn embeddings for
DataNode, Block, and Job nodes, then classify ``placed_on`` edges as good or
bad placements. The heterogeneous convolutions use ``SAGEConv`` because it
supports bipartite message passing between different node types.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, HeteroConv, Linear, SAGEConv

logger = logging.getLogger(__name__)

NodeType = str
EdgeType = Tuple[str, str, str]
NodeFeatureDict = Dict[NodeType, torch.Tensor]
EdgeIndexDict = Dict[EdgeType, torch.Tensor]

NODE_TYPES: Tuple[NodeType, ...] = ("datanode", "block", "job")
PLACED_ON: EdgeType = ("block", "placed_on", "datanode")
REV_PLACED_ON: EdgeType = ("datanode", "rev_placed_on", "block")
ACCESSES: EdgeType = ("job", "accesses", "block")
REV_ACCESSES: EdgeType = ("block", "rev_accesses", "job")

MESSAGE_EDGE_TYPES: Tuple[EdgeType, ...] = (
    PLACED_ON,
    REV_PLACED_ON,
    ACCESSES,
    REV_ACCESSES,
)

REVERSE_EDGE_TYPES: Dict[EdgeType, EdgeType] = {
    PLACED_ON: REV_PLACED_ON,
    ACCESSES: REV_ACCESSES,
}


def add_missing_reverse_edges(edge_index_dict: EdgeIndexDict) -> EdgeIndexDict:
    """Return an edge dictionary with reverse bipartite relations included.

    Args:
        edge_index_dict: Mapping from heterogeneous edge types to edge indices.

    Returns:
        A shallow copy of ``edge_index_dict`` that includes reverse edge types
        for ``placed_on`` and ``accesses`` when they are not already present.
    """
    completed_edge_index_dict = dict(edge_index_dict)

    for forward_edge_type, reverse_edge_type in REVERSE_EDGE_TYPES.items():
        if forward_edge_type in completed_edge_index_dict:
            completed_edge_index_dict.setdefault(
                reverse_edge_type,
                completed_edge_index_dict[forward_edge_type].flip(0),
            )

    return completed_edge_index_dict


class HeterogeneousGNN(nn.Module):
    """Heterogeneous GraphSAGE model for data locality optimization.

    The model applies three heterogeneous GraphSAGE layers by default across
    Block-to-DataNode placement edges and Job-to-Block access edges. Reverse
    relations are used so every node type can receive messages, and a small
    MLP classifier predicts binary labels for ``placed_on`` edges.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggr: str = "sum",
    ) -> None:
        """Initialize the heterogeneous GNN.

        Args:
            hidden_channels: Size of the learned node embeddings.
            num_layers: Number of heterogeneous GraphSAGE layers.
            dropout: Dropout probability used between GNN layers and in the
                edge classifier.
            aggr: Aggregation strategy used by ``HeteroConv``.

        Raises:
            ValueError: If ``num_layers`` is less than one.
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggr = aggr

        self.convs = nn.ModuleList(
            [self._create_hetero_conv(hidden_channels, aggr) for _ in range(num_layers)]
        )
        self.batch_norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: BatchNorm(hidden_channels)
                        for node_type in NODE_TYPES
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.edge_classifier = nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, 32),
            nn.ReLU(),
            Linear(32, 2),
        )

        logger.info(
            "Initialized HeterogeneousGNN with %s layers and %s hidden channels.",
            num_layers,
            hidden_channels,
        )

    def _create_hetero_conv(self, hidden_channels: int, aggr: str) -> HeteroConv:
        """Create one heterogeneous GraphSAGE convolution layer.

        Args:
            hidden_channels: Output dimension for every relation-specific
                ``SAGEConv`` layer.
            aggr: Aggregation strategy used to combine relation outputs that
                target the same node type.

        Returns:
            A ``HeteroConv`` layer containing one bipartite ``SAGEConv`` per
            message-passing relation.
        """
        conv_dict = {
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in MESSAGE_EDGE_TYPES
        }
        return HeteroConv(conv_dict, aggr=aggr)

    def _validate_inputs(
        self,
        x_dict: NodeFeatureDict,
        edge_index_dict: EdgeIndexDict,
    ) -> None:
        """Validate that the graph contains the node and edge types required.

        Args:
            x_dict: Node feature tensors keyed by node type.
            edge_index_dict: Edge index tensors keyed by edge type.

        Raises:
            ValueError: If a required node type or forward edge type is missing.
        """
        missing_node_types = [node_type for node_type in NODE_TYPES if node_type not in x_dict]
        missing_edge_types = [
            edge_type
            for edge_type in (PLACED_ON, ACCESSES)
            if edge_type not in edge_index_dict
        ]

        if missing_node_types:
            raise ValueError(f"Missing node feature tensors for: {missing_node_types}")
        if missing_edge_types:
            raise ValueError(f"Missing edge index tensors for: {missing_edge_types}")

    def forward(
        self,
        x_dict: NodeFeatureDict,
        edge_index_dict: EdgeIndexDict,
    ) -> NodeFeatureDict:
        """Compute node embeddings for all heterogeneous node types.

        Args:
            x_dict: Node feature tensors keyed by ``datanode``, ``block``, and
                ``job``.
            edge_index_dict: Edge indices keyed by heterogeneous edge type. The
                forward ``placed_on`` and ``accesses`` relations are required;
                reverse relations are added automatically when missing.

        Returns:
            Updated node embeddings keyed by node type.
        """
        self._validate_inputs(x_dict, edge_index_dict)
        edge_index_dict = add_missing_reverse_edges(edge_index_dict)

        for layer_idx, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            x_dict = conv(x_dict, edge_index_dict)

            missing_outputs = [node_type for node_type in NODE_TYPES if node_type not in x_dict]
            if missing_outputs:
                raise ValueError(
                    "The heterogeneous convolution did not produce embeddings "
                    f"for node types: {missing_outputs}"
                )

            x_dict = {
                node_type: F.relu(batch_norm[node_type](x_dict[node_type]))
                for node_type in NODE_TYPES
            }

            if layer_idx < self.num_layers - 1:
                x_dict = {
                    node_type: F.dropout(
                        node_embeddings,
                        p=self.dropout,
                        training=self.training,
                    )
                    for node_type, node_embeddings in x_dict.items()
                }

        return x_dict

    def predict_edge_labels(
        self,
        x_dict: NodeFeatureDict,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict binary labels for ``block -> placed_on -> datanode`` edges.

        Args:
            x_dict: Node embeddings produced by ``forward``.
            edge_index: Edge index tensor with shape ``[2, num_edges]`` where
                row 0 contains block indices and row 1 contains DataNode
                indices.

        Returns:
            Logits with shape ``[num_edges, 2]`` for binary classification.
        """
        block_embeddings = x_dict["block"][edge_index[0]]
        datanode_embeddings = x_dict["datanode"][edge_index[1]]
        edge_features = torch.cat([block_embeddings, datanode_embeddings], dim=-1)

        return self.edge_classifier(edge_features)


class SimpleGNN(nn.Module):
    """Compact heterogeneous GraphSAGE model for edge classification.

    This model keeps the same bipartite message-passing pattern as
    ``HeterogeneousGNN`` while using a smaller edge classifier. It is useful for
    debugging and quick experiments.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggr: str = "mean",
    ) -> None:
        """Initialize the simplified heterogeneous GNN.

        Args:
            hidden_channels: Size of the learned node embeddings.
            num_layers: Number of heterogeneous GraphSAGE layers.
            dropout: Dropout probability applied between convolution layers.
            aggr: Aggregation strategy used by ``HeteroConv``.

        Raises:
            ValueError: If ``num_layers`` is less than one.
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggr = aggr

        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {
                        edge_type: SAGEConv((-1, -1), hidden_channels)
                        for edge_type in MESSAGE_EDGE_TYPES
                    },
                    aggr=aggr,
                )
                for _ in range(num_layers)
            ]
        )
        self.batch_norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: BatchNorm(hidden_channels)
                        for node_type in NODE_TYPES
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = Linear(hidden_channels * 2, 2)

    def _validate_inputs(
        self,
        x_dict: NodeFeatureDict,
        edge_index_dict: EdgeIndexDict,
    ) -> None:
        """Validate that the graph contains the node and edge types required.

        Args:
            x_dict: Node feature tensors keyed by node type.
            edge_index_dict: Edge index tensors keyed by edge type.

        Raises:
            ValueError: If a required node type or forward edge type is missing.
        """
        missing_node_types = [node_type for node_type in NODE_TYPES if node_type not in x_dict]
        missing_edge_types = [
            edge_type
            for edge_type in (PLACED_ON, ACCESSES)
            if edge_type not in edge_index_dict
        ]

        if missing_node_types:
            raise ValueError(f"Missing node feature tensors for: {missing_node_types}")
        if missing_edge_types:
            raise ValueError(f"Missing edge index tensors for: {missing_edge_types}")

    def forward(
        self,
        x_dict: NodeFeatureDict,
        edge_index_dict: EdgeIndexDict,
    ) -> NodeFeatureDict:
        """Compute node embeddings with stacked heterogeneous GraphSAGE layers.

        Args:
            x_dict: Node feature tensors keyed by node type.
            edge_index_dict: Edge indices keyed by heterogeneous edge type. The
                reverse relations are added automatically when missing.

        Returns:
            Updated node embeddings keyed by node type.
        """
        self._validate_inputs(x_dict, edge_index_dict)
        edge_index_dict = add_missing_reverse_edges(edge_index_dict)

        for layer_idx, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            x_dict = conv(x_dict, edge_index_dict)

            missing_outputs = [node_type for node_type in NODE_TYPES if node_type not in x_dict]
            if missing_outputs:
                raise ValueError(
                    "The heterogeneous convolution did not produce embeddings "
                    f"for node types: {missing_outputs}"
                )

            x_dict = {
                node_type: F.relu(batch_norm[node_type](x_dict[node_type]))
                for node_type in NODE_TYPES
            }

            if layer_idx < self.num_layers - 1:
                x_dict = {
                    node_type: F.dropout(
                        node_embeddings,
                        p=self.dropout,
                        training=self.training,
                    )
                    for node_type, node_embeddings in x_dict.items()
                }

        return x_dict

    def predict_edge_labels(
        self,
        x_dict: NodeFeatureDict,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict binary labels for ``block -> placed_on -> datanode`` edges.

        Args:
            x_dict: Node embeddings produced by ``forward``.
            edge_index: Edge index tensor with block indices in row 0 and
                DataNode indices in row 1.

        Returns:
            Logits with shape ``[num_edges, 2]`` for binary classification.
        """
        block_embeddings = x_dict["block"][edge_index[0]]
        datanode_embeddings = x_dict["datanode"][edge_index[1]]
        edge_features = torch.cat([block_embeddings, datanode_embeddings], dim=-1)

        return self.classifier(edge_features)


def build_synthetic_inputs() -> Tuple[NodeFeatureDict, EdgeIndexDict]:
    """Build a small synthetic heterogeneous graph for smoke testing.

    Returns:
        A tuple containing node features and edge indices with the same node
        and edge types as the production graph.
    """
    torch.manual_seed(42)

    num_datanodes = 50
    num_blocks = 1_000
    num_jobs = 500
    num_placements = 2_000
    num_accesses = 3_000

    x_dict = {
        "datanode": torch.randn(num_datanodes, 5),
        "block": torch.randn(num_blocks, 4),
        "job": torch.randn(num_jobs, 6),
    }

    placed_on_edges = torch.stack(
        [
            torch.randint(0, num_blocks, (num_placements,)),
            torch.randint(0, num_datanodes, (num_placements,)),
        ],
        dim=0,
    )
    access_edges = torch.stack(
        [
            torch.randint(0, num_jobs, (num_accesses,)),
            torch.randint(0, num_blocks, (num_accesses,)),
        ],
        dim=0,
    )

    edge_index_dict = {
        PLACED_ON: placed_on_edges,
        ACCESSES: access_edges,
    }

    return x_dict, edge_index_dict


def test_model() -> None:
    """Run a smoke test for both heterogeneous GNN implementations.

    The test verifies that GraphSAGE bipartite message passing produces one
    hidden embedding per node and two logits per ``placed_on`` edge.
    """
    x_dict, edge_index_dict = build_synthetic_inputs()

    model = HeterogeneousGNN(hidden_channels=64, num_layers=3, dropout=0.3)
    model.eval()

    with torch.no_grad():
        embeddings = model(x_dict, edge_index_dict)
        logits = model.predict_edge_labels(embeddings, edge_index_dict[PLACED_ON])

    assert embeddings["datanode"].shape == (50, 64)
    assert embeddings["block"].shape == (1_000, 64)
    assert embeddings["job"].shape == (500, 64)
    assert logits.shape == (edge_index_dict[PLACED_ON].size(1), 2)

    simple_model = SimpleGNN(hidden_channels=64, num_layers=3, dropout=0.3)
    simple_model.eval()

    with torch.no_grad():
        simple_embeddings = simple_model(x_dict, edge_index_dict)
        simple_logits = simple_model.predict_edge_labels(
            simple_embeddings,
            edge_index_dict[PLACED_ON],
        )

    assert simple_embeddings["datanode"].shape == (50, 64)
    assert simple_embeddings["block"].shape == (1_000, 64)
    assert simple_embeddings["job"].shape == (500, 64)
    assert simple_logits.shape == (edge_index_dict[PLACED_ON].size(1), 2)

    print("HeterogeneousGNN output shapes:")
    print(f"  datanode: {tuple(embeddings['datanode'].shape)}")
    print(f"  block: {tuple(embeddings['block'].shape)}")
    print(f"  job: {tuple(embeddings['job'].shape)}")
    print(f"  placed_on logits: {tuple(logits.shape)}")
    print("SimpleGNN output shapes:")
    print(f"  datanode: {tuple(simple_embeddings['datanode'].shape)}")
    print(f"  block: {tuple(simple_embeddings['block'].shape)}")
    print(f"  job: {tuple(simple_embeddings['job'].shape)}")
    print(f"  placed_on logits: {tuple(simple_logits.shape)}")
    print("Model smoke test passed.")


if __name__ == "__main__":
    test_model()
