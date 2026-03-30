from pathlib import Path

import pandas as pd
import pytest

from tpx3awkward import convert_tpx3_file, convert_tpx3_files

DATA_DIR = Path(__file__).parents[1] / "data"


@pytest.mark.filterwarnings("ignore")
def test_convert_tpx3_file(tmp_path):
    path_to_data = DATA_DIR / "raw_test_data_0.tpx3"
    convert_tpx3_file(path_to_data, output_dir=tmp_path, print_details=True)

    cdf = pd.read_parquet(tmp_path / "raw_test_data_0_cent.parquet")
    required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n"}
    assert required.issubset(cdf.columns)


@pytest.mark.filterwarnings("ignore")
def test_convert_tpx3_files(tmp_path):
    path_to_data = DATA_DIR
    raw_tpx3_file_paths = [p for p in Path(path_to_data).rglob("*") if p.is_file()]
    convert_tpx3_files(raw_tpx3_file_paths, output_dir=tmp_path, print_details=True)

    for i in range(len(raw_tpx3_file_paths)):
        cdf = pd.read_parquet(tmp_path / f"raw_test_data_{i}_cent.parquet")
        required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n"}
        assert required.issubset(cdf.columns)
