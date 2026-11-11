"""Datasets: toy data for tests; BEIR download, embedding and caching for the Bench."""

from trivec.data.prepare import PreparedData, prepare
from trivec.data.toy import ToyDataset, l2_normalize, make_toy

__all__ = ["PreparedData", "ToyDataset", "l2_normalize", "make_toy", "prepare"]
