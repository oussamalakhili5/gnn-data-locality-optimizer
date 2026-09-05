# Architecture du Projet

## Vue Globale

\```
+----------------+     +----------------+     +----------------+
| Data Collection | --> | HDFS Storage   | --> | Spark Processing|
+----------------+     +----------------+     +----------------+
                                                     |
                                                     v
+----------------+     +----------------+     +----------------+
| GNN Model      | <-- | Graph Builder  | <-- | Features       |
+----------------+     +----------------+     +----------------+
        |
        v
+----------------+
| Optimized      |
| Placement      |
+----------------+
\```

## Composants

1. **DataCollector** : génère les données synthétiques réalistes.
2. **HDFS** : stocke les données brutes de manière distribuée.
3. **SparkProcessor** : agrège et nettoie les données à grande échelle.
4. **GraphBuilder** : transforme les données en graphe hétérogène PyG.
5. **HeterogeneousGNN** : apprend les embeddings des nœuds et prédit les bons placements.
6. **Evaluator** : mesure les performances et produit les visualisations.

## Types de nœuds et arêtes

- Nœuds : `datanode`, `block`, `job`
- Arêtes : `(block, placed_on, datanode)`, `(job, accesses, block)`