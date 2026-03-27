from pathlib import Path
import pytest
import pandas as pd
import time

from tpx3awkward.processing.cluster import (
    DEFAULT_CLUSTER_RADIUS,
    DEFAULT_CLUSTER_TW,
    cluster_raw_df
)
from tpx3awkward import tpx_to_raw_df, convert_tpx3_files_parallel, convert_tpx3_file
from tpx3awkward.processing.files import find_unmatched_tpx3_files, f_type
from tpx3awkward.processing.pipeline import drop_zero_tot


test_folder = Path(__file__).parents[1] / "data"
test_target = test_folder / "raw_test_data.tpx3"
target_out_pq = test_folder / "raw_test_data_cent.parquet"
target_out_hdf = test_folder / "tests/data/raw_test_data.h5"


class TestIntegration:
    def clear_dir(self):
        for path in test_folder.iterdir():
            if path.is_file() and path.suffix in (".parquet", "h5"):
                path.unlink(missing_ok=True)

    def test_automation_parallel(self):
        fpaths = find_unmatched_tpx3_files(directory_list=[test_folder])
        convert_tpx3_files_parallel(fpaths=fpaths, num_workers=2, extension=".parquet", timewalk_correct=True)
        ex_tree = [Path(str(path).replace(".tpx3", "_cent.parquet")).exists() for path in fpaths]
        self.clear_dir()
        assert all(ex_tree)

    def test_transcription_full(self):
        convert_tpx3_file(test_target, extension=f_type.PARQUET, timewalk_correct=True)
        df = drop_zero_tot(tpx_to_raw_df(test_target))
        cdf = cluster_raw_df(df, DEFAULT_CLUSTER_TW, DEFAULT_CLUSTER_RADIUS, timewalk_correct=True)
        retrieve = pd.read_parquet(target_out_pq)
        self.clear_dir()
        assert len(retrieve) == len(cdf)
