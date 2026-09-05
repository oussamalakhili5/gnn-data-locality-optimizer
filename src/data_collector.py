"""
Module de génération de données synthétiques pour simuler un cluster distribué.
Simule des DataNodes, Blocks et Jobs avec des distributions réalistes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedSystemGenerator:
    """
    Génère des données synthétiques réalistes simulant un cluster Hadoop/Spark.
    """
    
    def __init__(
        self,
        num_datanodes: int = 50,
        num_blocks: int = 1000,
        num_jobs: int = 500,
        random_seed: int = 42,
        locality_threshold: float = 0.7,
    ):
        """
        Initialise le générateur avec les paramètres du cluster.
        
        Args:
            num_datanodes: Nombre de serveurs de stockage
            num_blocks: Nombre de blocs de données
            num_jobs: Nombre de tâches de traitement
            random_seed: Seed pour la reproductibilité
        """
        self.num_datanodes = num_datanodes
        self.num_blocks = num_blocks
        self.num_jobs = num_jobs
        self.random_seed = random_seed
        if not 0.0 <= locality_threshold <= 1.0:
            raise ValueError("locality_threshold must be between 0 and 1")
        self.locality_threshold = locality_threshold
        
        np.random.seed(random_seed)
        logger.info(f"✅ Générateur initialisé avec seed={random_seed}")
    
    def generate_datanodes(self) -> pd.DataFrame:
        """
        Génère les DataNodes (serveurs de stockage).
        
        Returns:
            DataFrame avec les caractéristiques des DataNodes
        """
        logger.info("🔄 Génération des DataNodes...")
        
        n = self.num_datanodes
        
        datanodes = pd.DataFrame({
            'node_id': range(n),
            # Nom d'hôte
            'hostname': [f'datanode-{i:03d}.cluster.local' for i in range(n)],
            # Capacité de stockage en TB (distribution log-normale)
            'capacity_tb': np.random.lognormal(mean=1.5, sigma=0.5, size=n),
            # Position dans le rack (1 à 10)
            'rack_id': np.random.randint(1, 11, n),
            # Bande passante en Gbps
            'bandwidth_gbps': np.random.lognormal(mean=0.5, sigma=0.8, size=n),
            # Charge CPU (0 à 1, distribution beta)
            'cpu_usage': np.random.beta(a=2, b=3, size=n),
            # Charge mémoire (0 à 1)
            'memory_usage': np.random.beta(a=2, b=4, size=n),
            # Latence réseau en ms
            'network_latency_ms': np.random.exponential(scale=10, size=n),
            # Opérations I/O par seconde
            'io_operations_per_sec': np.random.poisson(lam=1000, size=n),
        })
        
        logger.info(f"✅ {n} DataNodes générés")
        return datanodes
    
    def generate_blocks(self) -> pd.DataFrame:
        """
        Génère les blocs de données (comme les blocs HDFS).
        
        Returns:
            DataFrame avec les caractéristiques des Blocks
        """
        logger.info("🔄 Génération des Blocks...")
        
        n = self.num_blocks
        
        blocks = pd.DataFrame({
            'block_id': range(n),
            # Chemin du fichier
            'file_path': [f'/data/file_{i:05d}.dat' for i in range(n)],
            # Taille en MB (autour de 128 MB)
            'size_mb': 128 * np.random.uniform(0.8, 1.2, n),
            # Popularité selon la loi de Zipf
            'popularity_score': np.random.zipf(1.5, n),
            # Fréquence d'accès (nombre de fois par jour)
            'access_frequency': np.random.poisson(lam=50, size=n),
            # Facteur de réplication (2, 3 ou 4 copies)
            'replication_factor': np.random.choice([2, 3, 4], n, p=[0.2, 0.6, 0.2]),
            # Date de création
            'creation_timestamp': pd.date_range('2024-01-01', periods=n, freq='10min'),
            # Dernier accès (heures depuis maintenant)
            'last_access_hours': np.random.exponential(24, n),
        })
        
        logger.info(f"✅ {n} Blocks générés")
        return blocks
    
    def generate_jobs(self) -> pd.DataFrame:
        """
        Génère les jobs (tâches de traitement Spark).
        
        Returns:
            DataFrame avec les caractéristiques des Jobs
        """
        logger.info("🔄 Génération des Jobs...")
        
        n = self.num_jobs
        
        jobs = pd.DataFrame({
            'job_id': range(n),
            # Nom du job
            'job_name': [f'spark-job-{i:04d}' for i in range(n)],
            # ID d'application
            'application_id': [f'app_{20240101}_{i:04d}' for i in range(n)],
            # Priorité
            'priority': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], n, p=[0.3, 0.5, 0.2]),
            # Durée en minutes
            'duration_minutes': np.random.lognormal(mean=3.0, sigma=1.0, size=n),
            # Ressources requises
            'executor_cores': np.random.randint(1, 16, n),
            'executor_memory_gb': np.random.lognormal(mean=2.0, sigma=0.8, size=n),
            # Données shuffle
            'shuffle_read_mb': np.random.lognormal(mean=5.0, sigma=1.5, size=n),
            'shuffle_write_mb': np.random.lognormal(mean=4.0, sigma=1.5, size=n),
            # Score de localité actuel (0 à 1)
            'data_locality_score': np.random.uniform(0.3, 0.9, n),
        })
        
        logger.info(f"✅ {n} Jobs générés")
        return jobs

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        """
        Normalize numeric values to the [0, 1] range.

        Args:
            values: Numeric array to normalize.

        Returns:
            Normalized NumPy array. Constant arrays are mapped to zeros.
        """
        values = np.asarray(values, dtype=float)
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)

        if np.isclose(max_value, min_value):
            return np.zeros_like(values, dtype=float)

        return (values - min_value) / (max_value - min_value)

    def _compute_datanode_quality(self, datanodes: pd.DataFrame) -> np.ndarray:
        """
        Compute a quality score for each DataNode from model-visible features.

        High-quality nodes have more capacity, more bandwidth, lower CPU and
        memory pressure, and lower network latency.

        Args:
            datanodes: DataFrame containing DataNode features.

        Returns:
            Array of quality scores in the [0, 1] range.
        """
        capacity_score = self._normalize(datanodes['capacity_tb'].values)
        bandwidth_score = self._normalize(np.log1p(datanodes['bandwidth_gbps'].values))
        latency_score = 1.0 - self._normalize(np.log1p(datanodes['network_latency_ms'].values))
        cpu_availability = 1.0 - datanodes['cpu_usage'].clip(0.0, 1.0).values
        memory_availability = 1.0 - datanodes['memory_usage'].clip(0.0, 1.0).values

        quality_score = (
            0.25 * capacity_score
            + 0.25 * bandwidth_score
            + 0.20 * latency_score
            + 0.15 * cpu_availability
            + 0.15 * memory_availability
        )

        return np.clip(quality_score, 0.0, 1.0)

    def _compute_block_demand(
        self,
        blocks: pd.DataFrame,
        access_patterns: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute a demand score for each block from model-visible signals.

        Popular and frequently accessed blocks should be placed on better
        DataNodes because poor placement creates more remote transfers.

        Args:
            blocks: DataFrame containing block features.
            access_patterns: DataFrame containing Job-to-Block accesses.

        Returns:
            Array of demand scores in the [0, 1] range.
        """
        block_ids = blocks['block_id'].values
        access_count_by_block = access_patterns.groupby('block_id')['access_count'].sum()
        transfer_by_block = access_patterns.groupby('block_id')['data_transferred_mb'].sum()

        access_counts = access_count_by_block.reindex(block_ids).fillna(0).values
        transferred_mb = transfer_by_block.reindex(block_ids).fillna(0.0).values

        popularity_score = self._normalize(np.log1p(blocks['popularity_score'].values))
        access_frequency_score = self._normalize(blocks['access_frequency'].values)
        observed_access_score = self._normalize(np.log1p(access_counts))
        transfer_score = self._normalize(np.log1p(transferred_mb))

        demand_score = (
            0.40 * popularity_score
            + 0.30 * access_frequency_score
            + 0.20 * observed_access_score
            + 0.10 * transfer_score
        )

        return np.clip(demand_score, 0.0, 1.0)
    
    def generate_access_patterns(self, blocks: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les patterns d'accès (quel job accède à quel bloc).
        
        Args:
            blocks: DataFrame des blocks
            jobs: DataFrame des jobs
            
        Returns:
            DataFrame avec les relations Job → Block
        """
        logger.info("🔄 Génération des patterns d'accès...")
        
        access_patterns = []
        
        # Biais vers les blocs populaires (loi de Zipf)
        block_weights = blocks['popularity_score'].values + 0.1
        block_weights = block_weights / block_weights.sum()
        
        for job_id in range(self.num_jobs):
            # Les jobs prioritaires accèdent à plus de blocs
            priority = jobs.loc[job_id, 'priority']
            if priority == 'HIGH':
                num_accesses = np.random.randint(10, 30)
            elif priority == 'MEDIUM':
                num_accesses = np.random.randint(5, 15)
            else:
                num_accesses = np.random.randint(1, 8)
            
            # Choisir les blocs accédés (biais vers les populaires)
            accessed_blocks = np.random.choice(
                blocks['block_id'].values,
                size=num_accesses,
                p=block_weights,
                replace=False
            )
            
            for block_id in accessed_blocks:
                access_patterns.append({
                    'job_id': job_id,
                    'block_id': int(block_id),
                    # Nombre d'accès
                    'access_count': np.random.poisson(lam=10),
                    # Volume de données transférées en MB
                    'data_transferred_mb': np.random.lognormal(mean=3.0, sigma=1.0),
                })
        
        access_df = pd.DataFrame(access_patterns)
        logger.info(f"✅ {len(access_df)} relations d'accès générées")
        return access_df
    
    def generate_placements(
        self,
        blocks: pd.DataFrame,
        datanodes: pd.DataFrame,
        access_patterns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Génère le placement initial des blocs sur les DataNodes.

        Labels are generated from a learnable placement policy instead of a
        random Bernoulli draw. A placement is local/good when its DataNode
        quality is in the top locality quantile for that block.
        
        Args:
            blocks: DataFrame des blocks
            datanodes: DataFrame des DataNodes
            access_patterns: DataFrame des patterns d'accès
            
        Returns:
            DataFrame avec les placements Block → DataNode
        """
        logger.info("🔄 Génération des placements initiaux...")
        
        placements = []
        datanode_quality = self._compute_datanode_quality(datanodes)
        block_demand = self._compute_block_demand(blocks, access_patterns)
        node_ids = datanodes['node_id'].values.astype(int)
        node_capacity_score = self._normalize(datanodes['capacity_tb'].values)
        node_bandwidth_score = self._normalize(np.log1p(datanodes['bandwidth_gbps'].values))
        block_size_score = self._normalize(blocks['size_mb'].values)

        for block_position, block in blocks.reset_index(drop=True).iterrows():
            demand = float(block_demand[block_position])
            size_pressure = float(block_size_score[block_position])

            # Hot blocks require a smaller pool of high-quality DataNodes.
            good_pool_fraction = float(np.clip(0.55 - 0.35 * demand, 0.18, 0.55))
            good_pool_size = int(np.clip(
                round(self.num_datanodes * good_pool_fraction),
                3,
                max(3, self.num_datanodes - 3)
            ))

            capacity_fit = 1.0 - np.maximum(0.0, size_pressure - node_capacity_score)
            compatibility = (
                0.70 * datanode_quality
                + 0.20 * capacity_fit
                + 0.10 * node_bandwidth_score
            )
            locality_cutoff = np.quantile(
                compatibility,
                self.locality_threshold,
            )

            ranked_nodes = np.argsort(compatibility)
            good_nodes = ranked_nodes[-good_pool_size:]
            bad_nodes = ranked_nodes[:-good_pool_size]
            selected_nodes = set()

            for replica in range(block['replication_factor']):
                target_local_probability = 0.30 + 0.45 * demand
                should_choose_local = np.random.random() < target_local_probability
                candidate_pool = good_nodes if should_choose_local else bad_nodes
                fallback_pool = bad_nodes if should_choose_local else good_nodes

                available_candidates = [
                    int(node_idx)
                    for node_idx in candidate_pool
                    if int(node_idx) not in selected_nodes
                ]
                if not available_candidates:
                    available_candidates = [
                        int(node_idx)
                        for node_idx in fallback_pool
                        if int(node_idx) not in selected_nodes
                    ]
                if not available_candidates:
                    available_candidates = [
                        int(node_idx)
                        for node_idx in range(self.num_datanodes)
                    ]

                if should_choose_local:
                    weights = compatibility[available_candidates] + 1e-6
                else:
                    weights = (1.0 - compatibility[available_candidates]) + 1e-6
                weights = weights / weights.sum()

                datanode_index = int(np.random.choice(available_candidates, p=weights))
                selected_nodes.add(datanode_index)
                is_local = compatibility[datanode_index] >= locality_cutoff

                placements.append({
                    'block_id': block['block_id'],
                    'datanode_id': int(node_ids[datanode_index]),
                    'replica_id': replica,
                    'locality_score': float(compatibility[datanode_index]),
                    'block_demand_score': demand,
                    'is_local': bool(is_local),
                })
        
        placements_df = pd.DataFrame(placements)
        logger.info(f"✅ {len(placements_df)} placements générés")
        logger.info(
            "✅ Locality labels: %.1f%% local, %.1f%% remote",
            100.0 * placements_df['is_local'].mean(),
            100.0 * (1.0 - placements_df['is_local'].mean())
        )
        return placements_df
    
    def generate_full_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Génère le dataset complet.
        
        Returns:
            Tuple de (datanodes, blocks, jobs, access_patterns, placements)
        """
        logger.info("🚀 Génération du dataset complet...")
        
        datanodes = self.generate_datanodes()
        blocks = self.generate_blocks()
        jobs = self.generate_jobs()
        access_patterns = self.generate_access_patterns(blocks, jobs)
        placements = self.generate_placements(blocks, datanodes, access_patterns)
        
        logger.info("✅ Dataset complet généré avec succès !")
        logger.info(f"   - {len(datanodes)} DataNodes")
        logger.info(f"   - {len(blocks)} Blocks")
        logger.info(f"   - {len(jobs)} Jobs")
        logger.info(f"   - {len(access_patterns)} accès")
        logger.info(f"   - {len(placements)} placements")
        
        return datanodes, blocks, jobs, access_patterns, placements
    
    def save_dataset(
        self,
        datanodes: pd.DataFrame,
        blocks: pd.DataFrame,
        jobs: pd.DataFrame,
        access_patterns: pd.DataFrame,
        placements: pd.DataFrame,
        output_dir: str = "data/raw"
    ) -> None:
        """
        Sauvegarde le dataset dans des fichiers CSV.
        
        Args:
            datanodes: DataFrame des DataNodes
            blocks: DataFrame des Blocks
            jobs: DataFrame des Jobs
            access_patterns: DataFrame des patterns d'accès
            placements: DataFrame des placements
            output_dir: Dossier de sortie
        """
        logger.info(f"💾 Sauvegarde des données dans {output_dir}...")
        
        # Créer le dossier s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder chaque DataFrame
        datanodes.to_csv(f"{output_dir}/datanodes.csv", index=False)
        blocks.to_csv(f"{output_dir}/blocks.csv", index=False)
        jobs.to_csv(f"{output_dir}/jobs.csv", index=False)
        access_patterns.to_csv(f"{output_dir}/access_patterns.csv", index=False)
        placements.to_csv(f"{output_dir}/placements.csv", index=False)
        
        logger.info(f"✅ Données sauvegardées dans {output_dir}/")
        logger.info(f"   - datanodes.csv ({len(datanodes)} lignes)")
        logger.info(f"   - blocks.csv ({len(blocks)} lignes)")
        logger.info(f"   - jobs.csv ({len(jobs)} lignes)")
        logger.info(f"   - access_patterns.csv ({len(access_patterns)} lignes)")
        logger.info(f"   - placements.csv ({len(placements)} lignes)")


# ============================================
# FONCTION PRINCIPALE
# ============================================

if __name__ == "__main__":
    # Créer le générateur
    generator = DistributedSystemGenerator(
        num_datanodes=50,
        num_blocks=1000,
        num_jobs=500,
        random_seed=42
    )
    
    # Générer le dataset complet
    datanodes, blocks, jobs, access_patterns, placements = generator.generate_full_dataset()
    
    # Sauvegarder les données
    generator.save_dataset(datanodes, blocks, jobs, access_patterns, placements)
    
    # Afficher un aperçu
    print("\n" + "="*60)
    print("📊 APERÇU DES DONNÉES GÉNÉRÉES")
    print("="*60)
    
    print("\n📦 DATANODES (5 premiers) :")
    print(datanodes[['node_id', 'capacity_tb', 'cpu_usage', 'bandwidth_gbps']].head())
    
    print("\n📦 BLOCKS (5 premiers) :")
    print(blocks[['block_id', 'size_mb', 'popularity_score', 'replication_factor']].head())
    
    print("\n📦 JOBS (5 premiers) :")
    print(jobs[['job_id', 'priority', 'duration_minutes', 'executor_cores']].head())
    
    print("\n📦 PATTERNS D'ACCÈS (5 premiers) :")
    print(access_patterns.head())
    
    print("\n📦 PLACEMENTS (5 premiers) :")
    print(placements.head())
