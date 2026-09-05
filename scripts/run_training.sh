#!/bin/bash
# ============================================
# Script d'entraînement complet
# ============================================

echo "🚀 Lancement de l'entraînement..."

# 1. Générer les données
echo "📊 Génération des données..."
python src/data_collector.py

# 2. Construire le graphe
echo "🏗️ Construction du graphe..."
python src/graph_builder.py

# 3. Entraîner le modèle
echo "🎯 Entraînement du GNN..."
python src/trainer.py

# 4. Évaluer
echo "📈 Évaluation..."
python src/evaluator.py

echo "✅ Pipeline terminé !"