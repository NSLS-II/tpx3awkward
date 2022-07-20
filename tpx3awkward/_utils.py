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
    2: photon event (id'd via 0xb upper nibble)
    3: TDC timstamp (id'd via 0x6 upper nibble)
    4: global timestap (id'd via 0x4 upper nibble)
    5: "command" data (id'd via 0x7 upper nibble)

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

    return output


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

    # drop the mysterious "command" lines and global timestamps for now
    ctypes = types[~((types == 5) | (types == 4))]
    cdata = data[~((types == 5) | (types == 4))]

    # find all of the packet headers
    packet_header_indx = np.where(ctypes == 1)[0]

    # sort out how many photons we have
    total_photons = np.sum(ctypes == 2)

    # allocate the return arrays
    x = np.zeros(total_photons, dtype="u4")
    y = np.zeros(total_photons, dtype="u4")
    pix_addr = np.zeros(total_photons, dtype="u4")
    ToA = np.zeros(total_photons, dtype="u4")
    ToT = np.zeros(total_photons, dtype="u4")
    FToA = np.zeros(total_photons, dtype="u4")
    SPIDR = np.zeros(total_photons, dtype="u4")
    chip_number = np.zeros(total_photons, dtype="u1")

    offset = 0
    # loop over the packet headers (can not vectorize this with numpy)
    for j in range(len(packet_header_indx)):
        # get the indexs we need into (cleaned) input data
        header_indx = packet_header_indx[j]
        first_photon = header_indx + 1

        # extract scalar information from the header
        num_pixels = int(get_block(cdata[header_indx], 16, 48) // 8)
        chip = int(get_block(cdata[header_indx], 8, 32))

        # grab what _should_ be the number of photons
        slc = slice(int(first_photon), int(first_photon + num_pixels))
        run_types = ctypes[slc]
        photons = cdata[slc]

        # but deal with batches that are short photons!!
        if not np.all(run_types == 2):
            # sometimes not enough photons come out, roll with it for now
            next_header = np.where(run_types != 2)[0][0]
            # print(f"{indx} {num_pixels} {np.where(run_types != 2)} has too few photons!")
            run_types = run_types[:next_header]
            photons = photons[:next_header]

        assert np.all(run_types == 2)

        num_photons = len(photons)
        # pixAddr is 16 bits, guess row-major
        pix_addr[offset : offset + num_photons] = (photons >> np.uint(44)) & np.uint(0xFFFF)
        x[offset : offset + num_photons] = pix_addr[offset : offset + num_photons] % 256
        y[offset : offset + num_photons] = pix_addr[offset : offset + num_photons] // 256
        # ToA is 14 bits
        ToA[offset : offset + num_photons] = (photons >> np.uint(30)) & np.uint(0x3FFF)
        # ToT is 10 bits
        ToT[offset : offset + num_photons] = (photons >> np.uint(20)) & np.uint(0x3FF)
        # FToA is 4 bits
        FToA[offset : offset + num_photons] = (photons >> np.uint(16)) & np.uint(0xF)
        # SPIDR time is 16 bits
        SPIDR[offset : offset + num_photons] = photons & np.uint(0xFFFF)
        # chip number (this is a constant)
        chip_number[offset : offset + num_photons] = chip
        # TODO also extract the original index (accounting for dropped values)

        offset += num_photons

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
