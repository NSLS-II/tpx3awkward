import pytest
from pathlib import Path

from tpx3awkward._utils import cluster, drop_zero_tot, process_raw_df, tpx_to_raw_df
from tpx3awkward._utils import DEFAULT_CLUSTER_TW, DEFAULT_CLUSTER_RADIUS
test_target = Path("tests/test_data/tpx_exp.tpx3")

class TestClustering:
    def test_stable_cnum(self):
        df = drop_zero_tot(tpx_to_raw_df(test_target))
        cdf = process_raw_df(df, DEFAULT_CLUSTER_TW, DEFAULT_CLUSTER_RADIUS, timewalk_correct=True)
        clust, events = cluster(df, DEFAULT_CLUSTER_TW, DEFAULT_CLUSTER_RADIUS)
        assert ((clust[-1]+1) == len(cdf))
