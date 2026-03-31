from pathlib import Path

from tpx3awkward.processing.cluster import DEFAULT_CLUSTER_RADIUS, DEFAULT_CLUSTER_TW, cluster, cluster_raw_df
from tpx3awkward.processing.decoding import (
    tpx_to_raw_df,
)
from tpx3awkward.processing.pipeline import drop_zero_tot

test_target = Path("tests/data/raw_test_data.tpx3")


class TestClustering:
    def test_stable_cnum(self):
        df = drop_zero_tot(tpx_to_raw_df(test_target))
        cdf = cluster_raw_df(df, DEFAULT_CLUSTER_TW, DEFAULT_CLUSTER_RADIUS, timewalk_correct=True)
        clust, _events = cluster(df, DEFAULT_CLUSTER_TW, DEFAULT_CLUSTER_RADIUS)
        assert (clust[-1] + 1) == len(cdf)
