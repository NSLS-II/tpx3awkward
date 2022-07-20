import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from typing import TypeVar, Union, Dict, List


IA = NDArray[np.uint64]
I = TypeVar("I", IA, np.uint64)


def raw_as_numpy(fpath: Union[str, Path]) -> I:
    """
    Read raw tpx3 data file as a numpy array.

    Each entry is read as a uint8 (64bit unsigned-integer)

    Parameters
    ----------

    """
    with open(fpath, "rb") as fin:
        return np.frombuffer(fin.read(), dtype="<u8")


def get_block(v: I, width: int, shift: int) -> I:
    return v >> np.uint64(shift) & np.uint64(2**width - 1)


def is_packet_header(v: I) -> I:
    return get_block(v, 32, 0) == 861425748


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


def ingest_raw_data(data: IA):
    types = classify_array(data)
    # drop the mysterious "command" lines and global timestamps for now
    ctypes = types[~((types == 5) | (types == 4))]
    cdata = data[~((types == 5) | (types == 4))]
    del data
    del types
    # find all of the packet headers
    (packet_header_indx,) = np.where(ctypes == 1)
    out = []
    # loop over the packet headers (can not vectorize this with numpy)
    for indx in packet_header_indx:
        num_pixels = int(get_block(cdata[indx], 16, 48) // 8)
        slc = slice(int(indx + 1), int(indx + num_pixels + 1))
        # grab what _should_S
        run_types = ctypes[slc]
        photons = cdata[slc]
        if not np.all(run_types == 2):
            # sometimes not enough photons come out, roll with it for now
            ((first, *rest),) = np.where(run_types != 2)
            print(f"{indx} {num_pixels} {np.where(run_types != 2)} has too few photons!")
            run_types = run_types[:first]
            photons = photons[:first]

        assert np.all(run_types == 2)
        # pixAddr is 16 bits
        pix_addr = (photons >> 44) & 0xFFFF
        # guessing at X / Y
        x = pix_addr // 256
        y = pix_addr % 256
        # ToA is 14 bits
        ToA = (photons >> 30) & 0x3FFF
        # ToT is 10 bits
        ToT = (photons >> 20) & 0x3FF
        # FToA is 4 bits
        FToA = (photons >> 16) & 0xF
        # SPIDR time is 16 bits
        spidr = photons & 0xFFFF

        out.append({"x": x, "y": y, "pix_addr": pix_addr, "ToA": ToA, "ToT": ToT, "FToA": FToA, "SPIDR": spidr})
    return out
