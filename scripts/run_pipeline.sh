#!/bin/bash
# ============================================================
# PIPELINE COMPLET : GNN Data Locality Optimizer
# Auteur : Oussama Lakhili
# Description :
#   Exécute séquentiellement :
#   1. Traitement Spark (optionnel, local ou HDFS)
#   2. Construction du graphe hétérogène
#   3. Entraînement du modèle GNN
#   4. Évaluation et génération des visualisations
# ============================================================

set -e  # Arrêter le script en cas d'erreur

echo "========================================="
echo " PIPELINE GNN DATA LOCALITY OPTIMIZER"
echo "========================================="

# ---- Étape 1 : Traitement Spark ----
# Par défaut : mode local. Utilisez --hdfs pour lire depuis HDFS Docker.
SPARK_MODE=${1:-local}   # "local" ou "hdfs"

echo ""
echo ">>> Étape 1 : Traitement Spark ($SPARK_MODE)"
if [ "$SPARK_MODE" = "hdfs" ]; then
    python src/spark_processor.py --hdfs
else
    python src/spark_processor.py
fi

# ---- Étape 2 : Construction du graphe ----
echo ""
echo ">>> Étape 2 : Construction du graphe"
python src/graph_builder.py

# ---- Étape 3 : Entraînement du GNN ----
echo ""
echo ">>> Étape 3 : Entraînement du modèle"
python src/trainer.py

# ---- Étape 4 : Évaluation ----
echo ""
echo ">>> Étape 4 : Évaluation"
python src/evaluator.py

echo ""
echo "========================================="
echo " PIPELINE TERMINÉ AVEC SUCCÈS"
echo "========================================="