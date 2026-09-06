"""
============================================================================
STREAMLIT APP : GNN Data Locality Optimizer
============================================================================
Author : Oussama Lakhili
Date : 2026-09-06
Description :
    Interactive web application to showcase the GNN model for data
    locality optimization. It loads the trained model and displays
    metrics, figures, and a simple prediction interface.
============================================================================
"""

# ============================================
# IMPORTS
# ============================================

import os
import sys
import torch
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from graph_builder import GraphBuilder
from gnn_model import HeterogeneousGNN
from trainer import prepare_graph_data, safe_torch_load, resolve_device

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="GNN Data Locality Optimizer",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 GNN Data Locality Optimizer")
st.markdown("Optimisation de la localité des données avec Graph Neural Networks")

# ============================================
# LOAD RESOURCES (CACHED)
# ============================================

@st.cache_resource
def load_resources():
    """
    Load the graph and the best model checkpoint.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    graph_path = os.path.join(project_root, "data", "graphs", "graph.pt")
    checkpoint_path = os.path.join(project_root, "data", "models", "best_model.pt")

    graph = safe_torch_load(graph_path, map_location="cpu")
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")

    model_cfg = checkpoint.get("model_config", {})
    model = HeterogeneousGNN(
        hidden_channels=model_cfg.get("hidden_channels", 64),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    return graph, model

graph, model = load_resources()

# ============================================
# SIDEBAR NAVIGATION
# ============================================

st.sidebar.header("Navigation")
page = st.sidebar.radio("Aller à", ["Accueil", "Résultats", "Prédiction"])

# ============================================
# PAGE ACCUEIL
# ============================================

if page == "Accueil":
    st.header("Présentation du projet")
    st.write(
        """
        Ce projet utilise un **Graph Neural Network hétérogène** pour prédire
        si un placement de bloc de données dans un cluster Hadoop/Spark est
        optimal (local) ou non (distant).
        """
    )
    st.image(
        "results/figures/confusion_matrix.png",
        caption="Matrice de confusion",
        use_container_width=True
    )

# ============================================
# PAGE RÉSULTATS
# ============================================

elif page == "Résultats":
    st.header("Métriques de performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "97.77%")
    col2.metric("Precision", "93.55%")
    col3.metric("Recall", "98.31%")
    col4.metric("F1-Score", "95.87%")

    st.image(
        "results/figures/learning_curves.png",
        caption="Courbes d'apprentissage",
        use_container_width=True
    )
    st.image(
        "results/figures/roc_curve.png",
        caption="Courbe ROC",
        use_container_width=True
    )

# ============================================
# PAGE PRÉDICTION
# ============================================
elif page == "Prédiction":
    st.header("Prédiction interactive")

    # Récupérer toutes les paires existantes
    edge_index = graph['block', 'placed_on', 'datanode'].edge_index
    valid_pairs = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    selected_pair = st.selectbox(
        "Choisissez une combinaison Block/DataNode",
        valid_pairs,
        format_func=lambda x: f"Block {x[0]} → DataNode {x[1]}"
    )

    if selected_pair:
        block_id, datanode_id = selected_pair
        st.write(f"Combinaison sélectionnée : Block {block_id}, DataNode {datanode_id}")

        if st.button("Prédire le placement"):
            data = prepare_graph_data(graph, torch.device("cpu"))
            with torch.no_grad():
                embeddings = model(data["x_dict"], data["edge_index_dict"])
                logits = model.predict_edge_labels(embeddings, data["edge_index"])
                mask = (edge_index[0] == block_id) & (edge_index[1] == datanode_id)
                idx = mask.nonzero(as_tuple=True)[0][0].item()
                prob_local = torch.softmax(logits[idx], dim=-1)[1].item()
                prediction = logits[idx].argmax().item()

            if prediction == 1:
                st.success(f"Placement prédit comme **LOCAL** avec une probabilité de {prob_local:.2%}")
            else:
                st.warning(f"Placement prédit comme **DISTANT** (probabilité d'être local : {prob_local:.2%})")
                
