from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tpx3awkward import convert_tpx3_file, convert_tpx3_files

DATA_DIR = Path(__file__).parents[1] / "data"
CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.mark.filterwarnings("ignore")
def test_convert_tpx3_file(tmp_path):
    path_to_data = DATA_DIR / "raw_test_data_00.tpx3"
    path_to_energy_parameters = CONFIG_DIR / "energy_estimation_test_parameters.npy"
    energy_estimation_test_parameters = np.load(path_to_energy_parameters)
    convert_tpx3_file(
        path_to_data,
        output_dir=tmp_path,
        energy_calib=energy_estimation_test_parameters,
        timewalk_correct=True,
        print_details=True,
    )

    cdf = pd.read_parquet(tmp_path / "raw_test_data_00_cent.parquet")
    required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n", "e_sum", "t_corr"}
    assert required.issubset(cdf.columns)


@pytest.mark.filterwarnings("ignore")
def test_convert_tpx3_files(tmp_path):
    path_to_data = DATA_DIR
    path_to_energy_parameters = CONFIG_DIR / "energy_estimation_test_parameters.npy"
    raw_tpx3_file_paths = sorted([p for p in Path(path_to_data).rglob("*") if p.is_file() and ".tpx3" in str(p)])
    convert_tpx3_files(
        raw_tpx3_file_paths,
        output_dir=tmp_path,
        timewalk_correct=True,
        energy_calib=path_to_energy_parameters,
        print_details=True,
    )

    for i in range(len(raw_tpx3_file_paths)):
        cdf = pd.read_parquet(tmp_path / f"raw_test_data_{i:02d}_cent.parquet")
        required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n", "e_sum", "t_corr"}
        assert required.issubset(cdf.columns)

    proc_parquet_files = sorted([p for p in Path(tmp_path).rglob("*") if p.is_file() and "cent.parquet" in str(p)])
    cur_proc_df = pd.concat([pd.read_parquet(f) for f in proc_parquet_files], ignore_index=True)

    # TODO remove this part when we are more confident in the refactor. This processing data is from the old tpx3awkward, and
    # is being used as a "stable" output to compare against.
    path_to_proc_data = DATA_DIR / "processed_concat_test_data.parquet"
    stable_proc_df = pd.read_parquet(path_to_proc_data)

    pd.testing.assert_frame_equal(cur_proc_df, stable_proc_df, atol=0.5)
