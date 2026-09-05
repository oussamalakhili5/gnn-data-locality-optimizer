"""
Unit tests for DataCollector.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_collector import DistributedSystemGenerator


def test_generate_full_dataset():
    """Check that the generated dataset has the expected sizes."""
    generator = DistributedSystemGenerator(
        num_datanodes=10,
        num_blocks=50,
        num_jobs=20,
        random_seed=42
    )
    datanodes, blocks, jobs, access_patterns, placements = generator.generate_full_dataset()

    assert len(datanodes) == 10
    assert len(blocks) == 50
    assert len(jobs) == 20
    assert len(access_patterns) > 0
    assert len(placements) > 0