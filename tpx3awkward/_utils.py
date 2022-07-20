import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from typing import TypeVar, Union, Dict
import numba

IA = NDArray[np.uint64]
UnSigned = TypeVar("I", IA, np.uint64)


def raw_as_numpy(fpath: Union[str, Path]) -> IA:
    """
    Read raw tpx3 data file as a numpy array.

    Each entry is read as a uint8 (64bit unsigned-integer)

    Parameters
    ----------

    """
    with open(fpath, "rb") as fin:
        return np.frombuffer(fin.read(), dtype="<u8")


@numba.jit(nopython=True)
def get_block(v: UnSigned, width: int, shift: int) -> UnSigned:
    return v >> np.uint64(shift) & np.uint64(2**width - 1)


@numba.jit(nopython=True)
def is_packet_header(v: UnSigned) -> UnSigned:
    return get_block(v, 32, 0) == 861425748


@numba.jit(nopython=True)
def classify_array(data: IA) -> NDArray[np.uint8]:
    """
    Create an array the same size as the data classifying 64bit uint by type.

    0: an unknown type (!!)
    1: packet header (id'd via TPX3 magic number)
    2: photon event (id'd via 0xB upper nibble)
    3: TDC timstamp (id'd via 0x6 upper nibble)
    4: global timestap (id'd via 0x4 upper nibble)
    5: "command" data (id'd via 0x7 upper nibble)
    6: frame driven data (id'd via 0xA upper nibble) (??)
    """
    output = np.zeros_like(data, dtype="<u1")
    # identify packet headers by magic number (TPX3 as ascii on lowest 8 bytes]
    is_header = is_packet_header(data)
    output[is_header] = 1
    # get the highest nibble
    nibble = data >> np.uint(60)
    # probably a better way to do this, but brute force!
    output[~is_header & (nibble == 0xB)] = 2
    output[~is_header & (nibble == 0x6)] = 3
    output[~is_header & (nibble == 0x4)] = 4
    output[~is_header & (nibble == 0x7)] = 5
    output[~is_header & (nibble == 0xA)] = 6

    return output


@numba.jit(nopython=True)
def _shift_xy(chip, row, col):
    # TODO sort out if this needs to be paremeterized
    out = np.zeros(2, "u4")
    if chip == 0:
        out[0] = row
        out[1] = col + np.uint(256)
    elif chip == 1:
        out[0] = np.uint(511) - row
        out[1] = np.uint(511) - col
    elif chip == 2:
        out[0] = np.uint(511) - row
        out[1] = np.uint(255) - col
    elif chip == 3:
        out[0] = row
        out[1] = col
    else:
        # TODO sort out how to get the chip number in here and make numba happy
        raise RuntimeError("Unknown chip id")
    return out


@numba.jit(nopython=True)
def _ingest_raw_data(data: IA):
    types = np.zeros_like(data, dtype="<u1")
    # identify packet headers by magic number (TPX3 as ascii on lowest 8 bytes]
    is_header = is_packet_header(data)
    types[is_header] = 1
    # get the highest nibble
    nibble = data >> np.uint(60)
    # probably a better way to do this, but brute force!
    types[~is_header & (nibble == 0xB)] = 2
    types[~is_header & (nibble == 0x6)] = 3
    types[~is_header & (nibble == 0x4)] = 4
    types[~is_header & (nibble == 0x7)] = 5

    # sort out how many photons we have
    total_photons = np.sum(types == 2)

    # allocate the return arrays
    x = np.zeros(total_photons, dtype="u4")
    y = np.zeros(total_photons, dtype="u4")
    pix_addr = np.zeros(total_photons, dtype="u2")
    ToA = np.zeros(total_photons, dtype="u4")
    ToT = np.zeros(total_photons, dtype="u4")
    FToA = np.zeros(total_photons, dtype="u4")
    SPIDR = np.zeros(total_photons, dtype="u4")
    chip_number = np.zeros(total_photons, dtype="u1")

    photon_offset = 0
    chip = np.uint(0)
    # expected_photon_count = np.uint(0)
    photon_run_count = 0
    # loop over the packet headers (can not vectorize this with numpy)
    for j in range(len(data)):
        msg = data[j]
        typ = types[j]
        if typ == 1:
            # if expected_photon_count != photon_run_count:
            #    print("missing photons!")
            # expected_photon_count = int(get_block(msg, 16, 48) // 8)
            # extract scalar information from the header

            chip = int(get_block(msg, 8, 32))
            photon_run_count = 0
        elif typ == 2:
            # pixAddr is 16 bits
            # these names and math are adapted from c++ code
            l_pix_addr = pix_addr[photon_offset] = (msg >> np.uint(44)) & np.uint(0xFFFF)
            # '1111111000000000'
            dcol = (l_pix_addr & np.uint(0xFE00)) >> np.uint(8)
            # '0000000111111000'
            spix = (l_pix_addr & np.uint(0x01F8)) >> np.uint(1)
            rowcol = _shift_xy(
                chip,
                # '0000000000000011'
                spix + (l_pix_addr & np.uint(0x3)),
                # '0000000000000100'
                dcol + ((l_pix_addr & np.uint(0x4)) >> np.uint(2)),
            )
            x[photon_offset] = rowcol[0]
            y[photon_offset] = rowcol[1]
            # ToA is 14 bits
            ToA[photon_offset] = (msg >> np.uint(30)) & np.uint(0x3FFF)
            # ToT is 10 bits
            ToT[photon_offset] = (msg >> np.uint(20)) & np.uint(0x3FF)
            # FToA is 4 bits
            FToA[photon_offset] = (msg >> np.uint(16)) & np.uint(0xF)
            # SPIDR time is 16 bits
            SPIDR[photon_offset] = msg & np.uint(0xFFFF)
            # chip number (this is a constant)
            chip_number[photon_offset] = chip
            photon_offset += 1
            photon_run_count += 1
        elif typ == 3:
            ...
        elif typ == 4:
            ...
        elif typ == 5:
            ...
        else:
            ...

    return x, y, pix_addr, ToA, ToT, FToA, SPIDR, chip_number


def ingest_raw_data(data: IA) -> Dict[str, NDArray]:
    """
    Parse values out of raw timepix3 data stream.

    Parameters
    ----------
    data : NDArray[np.unint64]
        The stream of raw data from the timepix3

    Returns
    -------
    Dict[str, NDArray]
       Keys of x, y, pix_addr, ToA, ToT, FToA, SPIDR, chip_number
    """
    return {
        k.strip(): v
        for k, v in zip("x, y, pix_addr, ToA, ToT, FToA, SPIDR, chip_number".split(","), _ingest_raw_data(data))
    }
