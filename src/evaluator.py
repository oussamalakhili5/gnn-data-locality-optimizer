"""
============================================================================
MODULE : Model Evaluation and Visualization
============================================================================
Author : Oussama Lakhili
Date : 2026-09-05
Description :
    Loads the best trained GNN model, evaluates it on the test set, and
    generates visualizations: confusion matrix, learning curves, ROC curve.
============================================================================
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

try:
    from .graph_builder import GraphBuilder
    from .gnn_model import HeterogeneousGNN
    from .trainer import prepare_graph_data, safe_torch_load, resolve_device
except ImportError:
    from graph_builder import GraphBuilder
    from gnn_model import HeterogeneousGNN
    from trainer import prepare_graph_data, safe_torch_load, resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluate a trained GNN model and produce visualizations.
    """

    def __init__(
        self,
        checkpoint_path: str = "data/models/best_model.pt",
        graph_path: str = "data/graphs/graph.pt",
        history_path: str = "data/models/training_history.json",
        output_dir: str = "results/figures",
        device: str = "auto",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.graph_path = Path(graph_path)
        self.history_path = Path(history_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(device)

        # Load graph
        logger.info("Loading graph from %s", self.graph_path)
        self.graph = safe_torch_load(self.graph_path, map_location="cpu")

        # Load checkpoint
        logger.info("Loading checkpoint from %s", self.checkpoint_path)
        self.checkpoint = safe_torch_load(self.checkpoint_path, map_location=self.device)

        # Recreate model architecture from checkpoint config
        model_cfg = self.checkpoint.get("model_config", {})
        self.model = HeterogeneousGNN(
            hidden_channels=model_cfg.get("hidden_channels", 64),
            num_layers=model_cfg.get("num_layers", 3),
            dropout=model_cfg.get("dropout", 0.3),
        ).to(self.device)

        # Load model weights
        state_dict = self.checkpoint.get("model_state_dict", self.checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("Model loaded and ready for evaluation")

        # Prepare graph data on device
        self.data = prepare_graph_data(self.graph, self.device)

        # Retrieve split masks from checkpoint if available
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        split_masks = self.checkpoint.get("split_masks")
        if split_masks:
            self.train_mask = split_masks.get("train")
            self.val_mask = split_masks.get("val")
            self.test_mask = split_masks.get("test")
            if self.train_mask is not None:
                self.train_mask = self.train_mask.to(self.device)
            if self.val_mask is not None:
                self.val_mask = self.val_mask.to(self.device)
            if self.test_mask is not None:
                self.test_mask = self.test_mask.to(self.device)
        else:
            logger.warning("No split masks found in checkpoint. Using all edges as test.")
            num_edges = self.data["edge_labels"].numel()
            self.test_mask = torch.ones(num_edges, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on all placed_on edges.

        Returns:
            Tuple (true_labels, predictions, positive_class_probabilities)
        """
        self.model.eval()
        embeddings = self.model(self.data["x_dict"], self.data["edge_index_dict"])
        logits = self.model.predict_edge_labels(embeddings, self.data["edge_index"])
        probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        predictions = logits.argmax(dim=-1).cpu().numpy()
        true_labels = self.data["edge_labels"].cpu().numpy()
        return true_labels, predictions, probabilities

    def evaluate_test(self) -> Dict[str, Any]:
        """Evaluate on the test set and return metrics."""
        true_all, pred_all, _ = self.predict()
        if self.test_mask is not None:
            mask = self.test_mask.cpu().numpy()
        else:
            mask = np.ones(len(true_all), dtype=bool)

        true = true_all[mask]
        pred = pred_all[mask]

        cm = confusion_matrix(true, pred, labels=[0, 1])
        report = classification_report(true, pred, output_dict=True, zero_division=0)

        metrics = {
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "test_accuracy": float(report["accuracy"]),
            "test_precision": float(report["1"]["precision"]),
            "test_recall": float(report["1"]["recall"]),
            "test_f1": float(report["1"]["f1-score"]),
        }
        logger.info("Test metrics: %s", metrics)
        return metrics

    def plot_confusion_matrix(self) -> None:
        """Plot and save the confusion matrix for the test set."""
        true_all, pred_all, _ = self.predict()
        if self.test_mask is not None:
            mask = self.test_mask.cpu().numpy()
            true = true_all[mask]
            pred = pred_all[mask]
        else:
            true = true_all
            pred = pred_all

        cm = confusion_matrix(true, pred, labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Remote (0)", "Local (1)"],
            yticklabels=["Remote (0)", "Local (1)"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix on Test Set")
        plt.tight_layout()
        save_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved confusion matrix to %s", save_path)

    def plot_learning_curves(self) -> None:
        """Plot and save training/validation learning curves."""
        if not self.history_path.exists():
            logger.warning("History file %s not found. Skipping learning curves.", self.history_path)
            return

        with open(self.history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        # The history may be nested under a "history" key
        hist_data = history.get("history", history)

        epochs = hist_data.get("epoch", [])
        train_loss = hist_data.get("train_loss", [])
        val_loss = hist_data.get("val_loss", [])
        train_acc = hist_data.get("train_accuracy", [])
        val_acc = hist_data.get("val_accuracy", [])
        train_f1 = hist_data.get("train_f1", [])
        val_f1 = hist_data.get("val_f1", [])

        if not epochs:
            logger.warning("No epoch data in history. Skipping learning curves.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(epochs, train_loss, label="Train")
        axes[0].plot(epochs, val_loss, label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curves")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, train_acc, label="Train")
        axes[1].plot(epochs, val_acc, label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Curves")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # F1 Score
        axes[2].plot(epochs, train_f1, label="Train")
        axes[2].plot(epochs, val_f1, label="Validation")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("F1 Score")
        axes[2].set_title("F1 Score Curves")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / "learning_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved learning curves to %s", save_path)

    def plot_roc_curve(self) -> None:
        """Plot and save the ROC curve for the test set."""
        true_all, _, prob_all = self.predict()
        if self.test_mask is not None:
            mask = self.test_mask.cpu().numpy()
            true = true_all[mask]
            prob = prob_all[mask]
        else:
            true = true_all
            prob = prob_all

        if len(np.unique(true)) < 2:
            logger.warning("Only one class present; ROC curve not plotted.")
            return

        fpr, tpr, _ = roc_curve(true, prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.output_dir / "roc_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved ROC curve to %s", save_path)

    def run_all(self) -> Dict[str, Any]:
        """Run all evaluations and save visualizations."""
        logger.info("Running evaluation...")
        metrics = self.evaluate_test()
        self.plot_confusion_matrix()
        self.plot_learning_curves()
        self.plot_roc_curve()
        logger.info("Evaluation complete. Figures saved to %s", self.output_dir)
        return metrics


def main() -> None:
    """Evaluate the model from the command line."""
    evaluator = Evaluator()
    metrics = evaluator.run_all()
    print("\nFinal Evaluation Metrics:")
    print(f"  Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"  Precision: {metrics['test_precision']:.4f}")
    print(f"  Recall:    {metrics['test_recall']:.4f}")
    print(f"  F1 Score:  {metrics['test_f1']:.4f}")


if __name__ == "__main__":
    main()