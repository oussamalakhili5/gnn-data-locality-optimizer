"""
============================================================================
MODULE : Spark Data Processor for Big Data Pipeline
============================================================================
Author : Oussama Lakhili
Date : 2026-09-05
Description :
    This module uses PySpark to read raw CSV data from HDFS (or local files)
    and perform large-scale aggregation and feature engineering before
    graph construction. It is designed to work with the Docker Hadoop/Spark
    cluster defined in docker/docker-compose.yml.
============================================================================
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


class SparkProcessor:
    """
    Process distributed system data using Apache Spark.
    """

    def __init__(
        self,
        app_name: str = "DataLocalityOptimization",
        master: Optional[str] = None,
        hdfs_enabled: bool = False,
        hdfs_host: str = "localhost",
        hdfs_port: int = 9000,
        data_dir: str = "data/raw",
    ):
        """
        Initialize the Spark processor.

        Args:
            app_name: Spark application name.
            master: Spark master URL. If None, local mode is used.
            hdfs_enabled: If True, read files from HDFS instead of local disk.
            hdfs_host: HDFS namenode hostname.
            hdfs_port: HDFS RPC port.
            data_dir: Local directory containing CSV files (used when
                `hdfs_enabled` is False).
        """
        self.hdfs_enabled = hdfs_enabled
        self.hdfs_host = hdfs_host
        self.hdfs_port = hdfs_port
        self.data_dir = data_dir

        # Create the Spark session
        builder = SparkSession.builder.appName(app_name)
        if master:
            builder = builder.master(master)

        # HDFS-specific configuration only when reading from HDFS
        if hdfs_enabled:
            builder = builder.config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
            builder = builder.config("spark.hadoop.dfs.datanode.use.datanode.hostname", "true")

        self.spark = builder.getOrCreate()

        logger.info("Spark session created (master=%s)", master or "local[*]")

    def _source_path(self, filename: str) -> str:
        """Return the absolute path for a dataset file."""
        if self.hdfs_enabled:
            return f"hdfs://{self.hdfs_host}:{self.hdfs_port}/data/raw/{filename}"
        return os.path.join(self.data_dir, filename)

    def read_csv(self, filename: str) -> DataFrame:
        """Read a CSV file into a Spark DataFrame."""
        path = self._source_path(filename)
        logger.info("Reading %s", path)
        return self.spark.read.option("header", True).option("inferSchema", True).csv(path)

    def aggregate_access_patterns(self, access_df: DataFrame) -> DataFrame:
        """
        Aggregate access patterns by job and block.

        Args:
            access_df: Raw access pattern DataFrame.

        Returns:
            Aggregated DataFrame with total accesses and data transferred.
        """
        logger.info("Aggregating access patterns...")
        aggregated = access_df.groupBy("job_id", "block_id").agg(
            F.sum("access_count").alias("total_accesses"),
            F.sum("data_transferred_mb").alias("total_data_transferred"),
            F.count("*").alias("access_events"),
        )
        return aggregated

    def compute_datanode_load(self, datanodes_df: DataFrame) -> DataFrame:
        """
        Compute average load metrics per rack.

        Args:
            datanodes_df: Raw DataNode DataFrame.

        Returns:
            Aggregated rack-level statistics.
        """
        logger.info("Computing DataNode load per rack...")
        rack_stats = datanodes_df.groupBy("rack_id").agg(
            F.avg("cpu_usage").alias("avg_cpu_load"),
            F.avg("memory_usage").alias("avg_memory_load"),
            F.avg("network_latency_ms").alias("avg_network_latency"),
        )
        return rack_stats

    def process_all(self) -> Dict[str, DataFrame]:
        """
        Run the complete Spark processing pipeline.

        Returns:
            Dictionary of processed Spark DataFrames.
        """
        logger.info("Starting Spark processing pipeline...")
        datanodes = self.read_csv("datanodes.csv")
        blocks = self.read_csv("blocks.csv")
        jobs = self.read_csv("jobs.csv")
        access_patterns = self.read_csv("access_patterns.csv")
        placements = self.read_csv("placements.csv")

        processed = {
            "datanodes": datanodes,
            "blocks": blocks,
            "jobs": jobs,
            "access_patterns": self.aggregate_access_patterns(access_patterns),
            "placements": placements,
            "rack_stats": self.compute_datanode_load(datanodes),
        }
        logger.info("Spark processing pipeline completed.")
        return processed

    def stop(self) -> None:
        """Stop the Spark session."""
        self.spark.stop()
        logger.info("Spark session stopped.")


def main() -> None:
    """Run the Spark processor from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Process data with Apache Spark.")
    parser.add_argument("--hdfs", action="store_true", help="Read data from HDFS")
    parser.add_argument("--master", default=None, help="Spark master URL")
    parser.add_argument("--data-dir", default="data/raw", help="Local data directory")
    args = parser.parse_args()

    processor = SparkProcessor(
        master=args.master,
        hdfs_enabled=args.hdfs,
        data_dir=args.data_dir,
    )
    processed = processor.process_all()
    for name, df in processed.items():
        logger.info("%s: %d rows", name, df.count())
    processor.stop()


if __name__ == "__main__":
    main()