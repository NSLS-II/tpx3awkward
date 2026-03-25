import pytest
from pathlib import Path
from tpx3awkward._utils import convert_tpx3_file, convert_tpx3_files_parallel, drop_zero_tot, find_unmatched_tpx3_files, process_raw_df, tpx_to_raw_df
from tpx3awkward._utils import DEFAULT_CLUSTER_RADIUS, DEFAULT_CLUSTER_TW
import pandas as pd

test_folder = Path("tests/data")
test_target = Path("tests/data/raw_test_data.tpx3")
target_out_pq = Path("tests/data/raw_test_data.parquet")
target_out_hdf = Path("tests/data/raw_test_data.h5")

class TestIntegration:
    def clear_dir(self):
        for path in test_folder.iterdir():
            if path.is_file() and path.suffix in (".parquet","h5"):
                path.unlink(missing_ok=True)


    def test_automation_parallel(self):
        fpaths = find_unmatched_tpx3_files(directory_list=[test_folder])
        convert_tpx3_files_parallel(fpaths=fpaths, num_workers=2, extension='.parquet', timewalk_correct=True)

        ex_tree = [Path(str(path).replace('.tpx','_cent.parquet')).exists() for path in fpaths]
        self.clear_dir()
        assert all(ex_tree)


    def test_transcription_full(self):
        convert_tpx3_file(test_target, extension='.parquet', timewalk_correct=True)
        df = drop_zero_tot(tpx_to_raw_df(test_target))
        cdf = process_raw_df(df, DEFAULT_CLUSTER_TW, DEFAULT_CLUSTER_RADIUS, timewalk_correct=True)
        retrieve = pd.read_parquet(target_out_pq)
        self.clear_dir()
        assert (len(retrieve) == len(cdf))
