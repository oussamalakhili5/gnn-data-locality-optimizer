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

param(
    [string]$Mode = "local"  # "local" ou "hdfs"
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " PIPELINE GNN DATA LOCALITY OPTIMIZER" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# ---- Étape 1 : Traitement Spark ----
Write-Host ""
Write-Host ">>> Étape 1 : Traitement Spark ($Mode)" -ForegroundColor Yellow
if ($Mode -eq "hdfs") {
    python src/spark_processor.py --hdfs
} else {
    python src/spark_processor.py
}

# ---- Étape 2 : Construction du graphe ----
Write-Host ""
Write-Host ">>> Étape 2 : Construction du graphe" -ForegroundColor Yellow
python src/graph_builder.py

# ---- Étape 3 : Entraînement du GNN ----
Write-Host ""
Write-Host ">>> Étape 3 : Entraînement du modèle" -ForegroundColor Yellow
python src/trainer.py

# ---- Étape 4 : Évaluation ----
Write-Host ""
Write-Host ">>> Étape 4 : Évaluation" -ForegroundColor Yellow
python src/evaluator.py

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host " PIPELINE TERMINÉ AVEC SUCCÈS" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green