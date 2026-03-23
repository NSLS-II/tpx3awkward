from pathlib import Path

import pandas as pd
import pytest

from tpx3awkward.processing import convert_tpx3_file

DATA_DIR = Path(__file__).parents[1] / "data"


@pytest.mark.filterwarnings("ignore")
def test_convert_tpx3_file(tmp_path):
    path_to_data = DATA_DIR / "raw_test_data.tpx3"
    convert_tpx3_file(path_to_data, output_dir=tmp_path, print_details=True)

    cdf = pd.read_parquet(tmp_path / "raw_test_data_cent.parquet")
    required = {"t", "xc", "yc", "ToT_max", "ToT_sum", "n"}
    assert required.issubset(cdf.columns)
