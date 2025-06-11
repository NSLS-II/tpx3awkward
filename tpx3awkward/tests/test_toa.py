from tpx3awkward._utils import get_block, matches_nibble, get_spidr, check_spidr_overflow

import numpy as np
import pytest


@pytest.fixture(scope="function")
def data(n=10):
    data = np.zeros(n, dtype=np.uint64)
    return data


@pytest.fixture(scope="function")
def header_msg(chip=3):
    return (np.uint8(chip) << np.uint(32)) + np.uint64(861425748)


@pytest.fixture(scope="function")
def empty_msg():
    return np.uint64(0xb) << np.uint(60)


def test_get_block(header_msg):
    assert np.uint8(get_block(header_msg, 8, 32)) == 3


def test_matches_nibble():
    msg = np.uint64(0xb) << np.uint(60)
    assert matches_nibble(msg, 0xb)


def test_get_spidr(empty_msg):
    msg1 = empty_msg + np.uint32(32768)
    msg2 = empty_msg + np.uint32(65535)
    assert get_spidr(msg1) == 32768
    assert get_spidr(msg2) == 65535

def test_check_spidr_overflow(empty_msg):
    spidr_arr = [0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10]
    spidr_arr.extend(range(10, 65535, 3))
    spidr_arr.extend([65534, 65534, 65535, 65534, 65534, 65535, 65535, 65534])
    data = [empty_msg + np.uint32(s) for s in spidr_arr]
    midpoint, last_spidr = check_spidr_overflow(data, 0)
    assert midpoint == 0
    assert last_spidr == 65535

    spidr_arr = list(range(20000, 65535))
    spidr_arr.extend(range(0, 10000))
    data = [empty_msg + np.uint32(s) for s in spidr_arr]
    midpoint, last_spidr = check_spidr_overflow(data, 0)
    assert midpoint == 14999
    assert last_spidr == 9999

    spidr_arr = [65534, 65534, 65535, 65534, 65534, 65535, 0, 65535, 1, 65534, 0, 0, 0, 1, 1, 2, 3, 4, 5]
    spidr_arr.extend(range(5, 2000))
    data = [empty_msg + np.uint32(s) for s in spidr_arr]
    midpoint, last_spidr = check_spidr_overflow(data, 0)
    assert midpoint == 33767
    assert last_spidr == 1999

    spidr_arr = list(range(20000, 65535))
    spidr_arr.extend([65534, 65534, 65535, 65534, 65534, 65535, 0, 65535, 1, 65534, 0, 0, 1, 1, 2, 3, 4, 5])

    data = [empty_msg + np.uint32(s) for s in spidr_arr]
    midpoint, last_spidr = check_spidr_overflow(data, 0)
    assert midpoint == 10002
    assert last_spidr == 5