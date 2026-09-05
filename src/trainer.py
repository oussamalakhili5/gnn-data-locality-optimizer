"""Training loop for the heterogeneous GraphSAGE edge classifier.

The trainer builds or loads the project graph, splits ``placed_on`` edges into
train/validation/test sets, trains ``HeterogeneousGNN`` with early stopping,
and saves a checkpoint that can be reused by the evaluator.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData

try:
    from .graph_builder import GraphBuilder
    from .gnn_model import ACCESSES, PLACED_ON, EdgeIndexDict, HeterogeneousGNN
except ImportError:
    from graph_builder import GraphBuilder
    from gnn_model import ACCESSES, PLACED_ON, EdgeIndexDict, HeterogeneousGNN

logger = logging.getLogger(__name__)

PreparedData = Dict[str, Any]
Metrics = Dict[str, Optional[float]]
SplitMasks = Dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training.

    Args:
        seed: Seed used by Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """Resolve a configured device string to a PyTorch device.

    Args:
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or any valid PyTorch device
            string.

    Returns:
        The selected ``torch.device``.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def safe_torch_load(path: Path | str, map_location: str | torch.device = "cpu") -> Any:
    """Load a local PyTorch artifact across PyTorch 2.x defaults.

    Args:
        path: Artifact path.
        map_location: Device mapping passed to ``torch.load``.

    Returns:
        The loaded object.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def json_default(value: Any) -> Any:
    """Convert tensors and NumPy values to JSON-serializable objects.

    Args:
        value: Object passed by ``json.dump``.

    Returns:
        A JSON-serializable representation.

    Raises:
        TypeError: If the value cannot be serialized.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(payload: Mapping[str, Any], path: Path | str) -> None:
    """Save a dictionary as formatted JSON.

    Args:
        payload: JSON payload.
        path: Destination path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)


def load_or_build_graph(
    data_dir: str = "data/raw",
    graph_path: Optional[str] = "data/graphs/graph.pt",
    rebuild_graph: bool = False,
) -> HeteroData:
    """Load a saved heterogeneous graph or rebuild it from raw CSV files.

    Args:
        data_dir: Directory containing the raw CSV files.
        graph_path: Optional path to a serialized ``HeteroData`` graph.
        rebuild_graph: Force rebuilding from raw CSV files when ``True``.

    Returns:
        The project heterogeneous graph.
    """
    graph_file = Path(graph_path) if graph_path else None

    if graph_file and graph_file.exists() and not rebuild_graph:
        logger.info("Loading graph from %s", graph_file)
        return safe_torch_load(graph_file, map_location="cpu")

    logger.info("Building graph from raw CSV files in %s", data_dir)
    return GraphBuilder(data_dir=data_dir).build_heterogeneous_graph()


def prepare_graph_data(graph: HeteroData, device: torch.device) -> PreparedData:
    """Move graph tensors to the selected device for training or evaluation.

    Args:
        graph: Heterogeneous graph produced by ``GraphBuilder``.
        device: Target device.

    Returns:
        Dictionary containing node features, message-passing edge indices,
        ``placed_on`` edge indices, and edge labels.

    Raises:
        ValueError: If required graph stores are missing.
    """
    for node_type in ("datanode", "block", "job"):
        if node_type not in graph.node_types:
            raise ValueError(f"Missing node type in graph: {node_type}")

    for edge_type in (PLACED_ON, ACCESSES):
        if edge_type not in graph.edge_types:
            raise ValueError(f"Missing edge type in graph: {edge_type}")

    placed_on_store = graph[PLACED_ON]
    if not hasattr(placed_on_store, "y"):
        raise ValueError("The placed_on edge store must contain binary labels in .y.")

    x_dict = {
        "datanode": graph["datanode"].x.to(device),
        "block": graph["block"].x.to(device),
        "job": graph["job"].x.to(device),
    }

    edge_index_dict: EdgeIndexDict = {
        edge_type: graph[edge_type].edge_index.to(device)
        for edge_type in graph.edge_types
    }

    edge_index = placed_on_store.edge_index.to(device)
    edge_labels = placed_on_store.y.long().to(device)

    return {
        "x_dict": x_dict,
        "edge_index_dict": edge_index_dict,
        "edge_index": edge_index,
        "edge_labels": edge_labels,
    }


def create_edge_split_masks(
    edge_labels: torch.Tensor,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create stratified train/validation/test masks for edge labels.

    Args:
        edge_labels: Binary labels for ``placed_on`` edges.
        train_ratio: Proportion of edges used for training.
        val_ratio: Proportion of edges used for validation.
        seed: Random seed used by the split.

    Returns:
        Boolean masks for train, validation, and test splits.

    Raises:
        ValueError: If the ratios are invalid.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    labels = edge_labels.detach().cpu().numpy()
    indices = np.arange(labels.shape[0])
    stratify = labels if len(np.unique(labels)) > 1 else None

    train_idx, remaining_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    remaining_labels = labels[remaining_idx]
    remaining_ratio = 1.0 - train_ratio
    relative_val_ratio = val_ratio / remaining_ratio
    remaining_stratify = (
        remaining_labels if len(np.unique(remaining_labels)) > 1 else None
    )

    val_idx, test_idx = train_test_split(
        remaining_idx,
        train_size=relative_val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=remaining_stratify,
    )

    train_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(labels.shape[0], dtype=torch.bool)

    train_mask[torch.from_numpy(train_idx)] = True
    val_mask[torch.from_numpy(val_idx)] = True
    test_mask[torch.from_numpy(test_idx)] = True

    return train_mask, val_mask, test_mask


def compute_class_weights(edge_labels: torch.Tensor) -> torch.Tensor:
    """Compute inverse-frequency class weights for binary labels.

    Args:
        edge_labels: Binary labels for ``placed_on`` edges.

    Returns:
        Tensor with one weight per class.
    """
    counts = torch.bincount(edge_labels.detach().cpu(), minlength=2).float()
    weights = edge_labels.numel() / (2.0 * counts.clamp_min(1.0))
    return weights.to(edge_labels.device)


def compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Metrics:
    """Compute binary classification metrics from logits and labels.

    Args:
        logits: Model outputs with shape ``[num_edges, 2]``.
        labels: Binary labels with shape ``[num_edges]``.

    Returns:
        Dictionary of scalar metrics.
    """
    if labels.numel() == 0:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "average_precision": None,
            "positive_rate": None,
            "tn": None,
            "fp": None,
            "fn": None,
            "tp": None,
        }

    probabilities = torch.softmax(logits.detach(), dim=-1)[:, 1].cpu().numpy()
    predictions = logits.detach().argmax(dim=-1).cpu().numpy()
    targets = labels.detach().cpu().numpy()
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics: Metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "positive_rate": float(targets.mean()),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }

    if len(np.unique(targets)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(targets, probabilities)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def format_metrics(metrics: Mapping[str, Optional[float]]) -> str:
    """Format the most important metrics for terminal logs.

    Args:
        metrics: Metrics dictionary.

    Returns:
        Compact formatted metric string.
    """
    parts = []
    for key in ("loss", "accuracy", "precision", "recall", "f1"):
        value = metrics.get(key)
        parts.append(f"{key}={value:.4f}" if value is not None else f"{key}=n/a")
    return " | ".join(parts)


class GNNTrainer:
    """Trainer for the heterogeneous GraphSAGE data-locality model.

    The class owns model initialization, stratified edge splitting, training,
    validation, early stopping, checkpointing, and final test evaluation.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 5e-4,
        epochs: int = 200,
        patience: int = 20,
        device: str = "auto",
        seed: int = 42,
        class_weighting: bool = True,
        checkpoint_path: str = "data/models/best_model.pt",
        history_path: str = "data/models/training_history.json",
        log_every: int = 10,
    ) -> None:
        """Initialize the trainer.

        Args:
            hidden_channels: Size of hidden node embeddings.
            num_layers: Number of heterogeneous GraphSAGE layers.
            dropout: Dropout probability.
            learning_rate: Optimizer learning rate.
            weight_decay: Adam weight decay.
            epochs: Maximum number of training epochs.
            patience: Early-stopping patience measured in epochs.
            device: ``"auto"``, ``"cpu"``, ``"cuda"``, or a valid device string.
            seed: Random seed for reproducible splits and initialization.
            class_weighting: Use inverse-frequency loss weights when ``True``.
            checkpoint_path: Destination for the best model checkpoint.
            history_path: Destination for the training history JSON file.
            log_every: Epoch logging interval.
        """
        set_seed(seed)

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.device = resolve_device(device)
        self.seed = seed
        self.class_weighting = class_weighting
        self.checkpoint_path = Path(checkpoint_path)
        self.history_path = Path(history_path)
        self.log_every = log_every
        self.train_ratio: Optional[float] = None
        self.val_ratio: Optional[float] = None

        self.model = HeterogeneousGNN(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: nn.Module = nn.CrossEntropyLoss()
        self.model_initialized = False

        self.train_mask: Optional[torch.Tensor] = None
        self.val_mask: Optional[torch.Tensor] = None
        self.test_mask: Optional[torch.Tensor] = None
        self.history: Dict[str, list[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_precision": [],
            "val_precision": [],
            "train_recall": [],
            "val_recall": [],
            "train_f1": [],
            "val_f1": [],
        }

        logger.info("Trainer initialized on device=%s", self.device)

    @property
    def model_config(self) -> Dict[str, Any]:
        """Return the model configuration used for checkpointing."""
        return {
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    @property
    def training_config(self) -> Dict[str, Any]:
        """Return the training configuration used for checkpointing."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "seed": self.seed,
            "class_weighting": self.class_weighting,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
        }

    def prepare_data(self, graph: HeteroData) -> PreparedData:
        """Prepare a graph for this trainer's device.

        Args:
            graph: Heterogeneous graph.

        Returns:
            Prepared graph tensors.
        """
        return prepare_graph_data(graph, self.device)

    def split_data(
        self,
        edge_labels_or_num_edges: torch.Tensor | int,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split edge labels into train, validation, and test masks.

        Args:
            edge_labels_or_num_edges: Edge labels. An integer is accepted for
                backward compatibility and creates a non-stratified fake label
                vector.
            train_ratio: Proportion of edges used for training.
            val_ratio: Proportion of edges used for validation.

        Returns:
            Boolean train, validation, and test masks.
        """
        if isinstance(edge_labels_or_num_edges, int):
            labels = torch.zeros(edge_labels_or_num_edges, dtype=torch.long)
        else:
            labels = edge_labels_or_num_edges.detach().cpu()

        masks = create_edge_split_masks(
            labels,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=self.seed,
        )
        logger.info(
            "Data split: %s train, %s validation, %s test edges.",
            int(masks[0].sum()),
            int(masks[1].sum()),
            int(masks[2].sum()),
        )
        return masks

    def initialize_model(self, data: PreparedData) -> None:
        """Run a lightweight forward pass to initialize lazy PyG parameters.

        Args:
            data: Prepared graph tensors.
        """
        if self.model_initialized:
            return

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data["x_dict"], data["edge_index_dict"])
            edge_index = data["edge_index"][:, :1]
            self.model.predict_edge_labels(embeddings, edge_index)

        self.model_initialized = True

    def configure_training(
        self,
        data: PreparedData,
        train_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Create the loss function and optimizer after lazy initialization.

        Args:
            data: Prepared graph tensors.
            train_mask: Optional mask used to compute class weights from the
                training split only.
        """
        self.initialize_model(data)

        if self.class_weighting:
            labels_for_weights = data["edge_labels"]
            if train_mask is not None:
                labels_for_weights = labels_for_weights[train_mask.to(self.device)]
            class_weights = compute_class_weights(labels_for_weights)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info("Using class weights: %s", class_weights.detach().cpu().tolist())
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def train_epoch(self, data: PreparedData, train_mask: torch.Tensor) -> Metrics:
        """Run one optimization epoch.

        Args:
            data: Prepared graph tensors.
            train_mask: Boolean mask selecting training edges.

        Returns:
            Training metrics for the selected edges.

        Raises:
            RuntimeError: If the optimizer has not been configured.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not configured. Call configure_training first.")

        self.model.train()
        self.optimizer.zero_grad()

        embeddings = self.model(data["x_dict"], data["edge_index_dict"])
        logits = self.model.predict_edge_labels(embeddings, data["edge_index"])
        mask = train_mask.to(self.device)

        loss = self.criterion(logits[mask], data["edge_labels"][mask])
        loss.backward()
        self.optimizer.step()

        metrics = compute_binary_metrics(logits[mask], data["edge_labels"][mask])
        metrics["loss"] = float(loss.item())
        return metrics

    @torch.no_grad()
    def evaluate(self, data: PreparedData, mask: torch.Tensor) -> Metrics:
        """Evaluate the model on a selected edge split.

        Args:
            data: Prepared graph tensors.
            mask: Boolean mask selecting evaluation edges.

        Returns:
            Evaluation metrics for the selected edges.
        """
        self.model.eval()

        embeddings = self.model(data["x_dict"], data["edge_index_dict"])
        logits = self.model.predict_edge_labels(embeddings, data["edge_index"])
        mask = mask.to(self.device)

        if int(mask.sum()) == 0:
            return compute_binary_metrics(logits[mask], data["edge_labels"][mask])

        loss = F.cross_entropy(logits[mask], data["edge_labels"][mask])
        metrics = compute_binary_metrics(logits[mask], data["edge_labels"][mask])
        metrics["loss"] = float(loss.item())
        return metrics

    def update_history(
        self,
        epoch: int,
        train_metrics: Mapping[str, Optional[float]],
        val_metrics: Mapping[str, Optional[float]],
    ) -> None:
        """Append one epoch of metrics to the in-memory history.

        Args:
            epoch: Current epoch number.
            train_metrics: Metrics computed on the training split.
            val_metrics: Metrics computed on the validation split.
        """
        self.history["epoch"].append(float(epoch))
        for key in ("loss", "accuracy", "precision", "recall", "f1"):
            self.history[f"train_{key}"].append(float(train_metrics.get(key) or 0.0))
            self.history[f"val_{key}"].append(float(val_metrics.get(key) or 0.0))

    def save_checkpoint(
        self,
        path: Path | str,
        epoch: int,
        best_val_metrics: Mapping[str, Optional[float]],
        test_metrics: Optional[Mapping[str, Optional[float]]] = None,
    ) -> None:
        """Save a checkpoint with model state, split masks, and metadata.

        Args:
            path: Destination checkpoint path.
            epoch: Best validation epoch.
            best_val_metrics: Metrics from the best validation epoch.
            test_metrics: Optional final test metrics.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "model_config": self.model_config,
            "training_config": self.training_config,
            "epoch": epoch,
            "best_val_metrics": dict(best_val_metrics),
            "test_metrics": dict(test_metrics) if test_metrics is not None else None,
            "history": self.history,
            "split_masks": {
                "train": self.train_mask.detach().cpu() if self.train_mask is not None else None,
                "val": self.val_mask.detach().cpu() if self.val_mask is not None else None,
                "test": self.test_mask.detach().cpu() if self.test_mask is not None else None,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved checkpoint to %s", checkpoint_path)

    def load_model(self, path: Path | str, data: Optional[PreparedData] = None) -> Dict[str, Any]:
        """Load model weights from a checkpoint.

        Args:
            path: Checkpoint path.
            data: Optional prepared data used to initialize lazy parameters
                before loading.

        Returns:
            Loaded checkpoint dictionary.
        """
        checkpoint = safe_torch_load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        if data is not None:
            self.initialize_model(data)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        return checkpoint

    def train(
        self,
        graph: HeteroData,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """Train the model and evaluate the best checkpoint on the test split.

        Args:
            graph: Heterogeneous graph.
            train_ratio: Proportion of ``placed_on`` edges used for training.
            val_ratio: Proportion of ``placed_on`` edges used for validation.

        Returns:
            Training summary containing history, best validation metrics, test
            metrics, and artifact paths.
        """
        logger.info("Starting training.")
        data = self.prepare_data(graph)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.train_mask, self.val_mask, self.test_mask = self.split_data(
            data["edge_labels"],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        self.configure_training(data, self.train_mask)

        best_epoch = 0
        best_val_f1 = -1.0
        best_val_metrics: Metrics = {}
        best_model_state: Optional[Dict[str, torch.Tensor]] = None
        best_optimizer_state: Optional[Dict[str, Any]] = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(data, self.train_mask)
            val_metrics = self.evaluate(data, self.val_mask)
            self.update_history(epoch, train_metrics, val_metrics)

            val_f1 = float(val_metrics.get("f1") or 0.0)
            improved = val_f1 > best_val_f1

            if improved:
                best_epoch = epoch
                best_val_f1 = val_f1
                best_val_metrics = dict(val_metrics)
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                best_optimizer_state = (
                    copy.deepcopy(self.optimizer.state_dict())
                    if self.optimizer is not None
                    else None
                )
                patience_counter = 0
            else:
                patience_counter += 1

            should_log = epoch == 1 or epoch % self.log_every == 0 or improved
            if should_log:
                logger.info(
                    "Epoch %03d/%03d | train: %s | val: %s",
                    epoch,
                    self.epochs,
                    format_metrics(train_metrics),
                    format_metrics(val_metrics),
                )

            if patience_counter >= self.patience:
                logger.info(
                    "Early stopping at epoch %s. Best epoch was %s.",
                    epoch,
                    best_epoch,
                )
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        if self.optimizer is not None and best_optimizer_state is not None:
            self.optimizer.load_state_dict(best_optimizer_state)

        test_metrics = self.evaluate(data, self.test_mask)
        self.save_checkpoint(
            self.checkpoint_path,
            epoch=best_epoch,
            best_val_metrics=best_val_metrics,
            test_metrics=test_metrics,
        )

        training_summary = {
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "checkpoint_path": str(self.checkpoint_path),
            "history_path": str(self.history_path),
            "history": self.history,
        }
        save_json(training_summary, self.history_path)

        logger.info("Training completed. Best validation F1: %.4f", best_val_f1)
        logger.info("Final test metrics: %s", format_metrics(test_metrics))
        return training_summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for training.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Train the heterogeneous GNN model.")
    parser.add_argument("--data-dir", default="data/raw", help="Directory with raw CSV files.")
    parser.add_argument(
        "--graph-path",
        default="data/graphs/graph.pt",
        help="Optional saved HeteroData graph path.",
    )
    parser.add_argument(
        "--rebuild-graph",
        action="store_true",
        help="Rebuild the graph from CSV files instead of loading graph-path.",
    )
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-class-weighting",
        action="store_true",
        help="Disable inverse-frequency class weights.",
    )
    parser.add_argument("--checkpoint-path", default="data/models/best_model.pt")
    parser.add_argument("--history-path", default="data/models/training_history.json")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main() -> None:
    """Train the model from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_arg_parser().parse_args()

    graph = load_or_build_graph(
        data_dir=args.data_dir,
        graph_path=args.graph_path,
        rebuild_graph=args.rebuild_graph,
    )

    trainer = GNNTrainer(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        class_weighting=not args.no_class_weighting,
        checkpoint_path=args.checkpoint_path,
        history_path=args.history_path,
        log_every=args.log_every,
    )
    summary = trainer.train(
        graph,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print("\nTraining completed")
    print(f"  Best epoch: {summary['best_epoch']}")
    print(f"  Checkpoint: {summary['checkpoint_path']}")
    print(f"  History: {summary['history_path']}")
    print(f"  Test: {format_metrics(summary['test_metrics'])}")


if __name__ == "__main__":
    main()
