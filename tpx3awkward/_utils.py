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
    basetime = np.zeros(total_photons, dtype="u8")
    timestamp = np.zeros(total_photons, dtype="u8")

    photon_offset = 0
    chip = np.uint16(0)
    expected_msg_count = np.uint16(0)
    msg_run_count = np.uint(0)

    heartbeat_lsb = np.uint64(0)
    heartbeat_msb = np.uint64(0)
    heartbeat_time = np.uint64(0)
    # loop over the packet headers (can not vectorize this with numpy)
    for j in range(len(data)):
        msg = data[j]
        typ = types[j]
        if typ == 1:
            # 1: packet header (id'd via TPX3 magic number)
            if expected_msg_count != msg_run_count:
                print("missing messages!", msg)
            # extract scalar information from the header

            # "number of pixels in chunk" is given in bytes not words
            # and means all words in the chunk, not just "photons"
            expected_msg_count = get_block(msg, 16, 48) // 8
            # what chip we are on
            chip = np.uint8(get_block(msg, 8, 32))
            msg_run_count = 0
        elif typ == 2 or typ == 6:
            #  2: photon event (id'd via 0xB upper nibble)
            #  6: frame driven data (id'd via 0xA upper nibble) (??)

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
            l_FToA = FToA[photon_offset] = (msg >> np.uint(16)) & np.uint(0xF)
            # SPIDR time is 16 bits
            SPIDR[photon_offset] = msg & np.uint(0xFFFF)
            # chip number (this is a constant)
            chip_number[photon_offset] = chip
            # heartbeat time
            basetime[photon_offset] = heartbeat_time

            ToA_coarse = (SPIDR[photon_offset] << np.uint(14)) | ToA[photon_offset]
            globaltime = (heartbeat_time & np.uint(0xFFFFC0000000)) | (ToA_coarse & np.uint(0x3FFFFFFF))
            # TODO the c++ code deals with jumps (presumable roll over?)
            # TODO the c++ code as shifts due to columns in the LSB
            timestamp[photon_offset] = ((globaltime << np.uint(12)) - (l_FToA << np.uint(8))) * 25
            photon_offset += 1
            msg_run_count += 1
        elif typ == 3:
            #  3: TDC timstamp (id'd via 0x6 upper nibble)
            # TODO: handle these!
            msg_run_count += 1
        elif typ == 4:
            #  4: global timestap (id'd via 0x4 upper nibble)
            subheader = (msg >> np.uint(56)) & np.uint(0x0F)
            if subheader == 0x4:
                # timer lsb, 32 bits of time
                heartbeat_lsb = (msg >> np.uint(16)) & np.uint(0xFFFFFFFF)
            elif subheader == 0x5:
                # timer msb

                time_msg = (msg >> np.uint(16)) & np.uint(0xFFFF)
                heartbeat_msb = time_msg << np.uint(32)
                # TODO the c++ code has large jump detection, do not understand why
                heartbeat_time = heartbeat_msb | heartbeat_lsb
            else:
                raise Exception("unknown header")

            msg_run_count += 1
        elif typ == 5:
            #  5: "command" data (id'd via 0x7 upper nibble)
            # TODO handle this!
            msg_run_count += 1
        else:
            raise Exception("Not supported")

    return x, y, pix_addr, ToA, ToT, FToA, SPIDR, chip_number, basetime, timestamp


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
        for k, v in zip(
            "x, y, pix_addr, ToA, ToT, FToA, SPIDR, chip_number, basetime, timestamp".split(","),
            _ingest_raw_data(data),
        )
    }


d = raw_as_numpy("/mnt/store/bnl/cache/chx_timepix/2022/7/18/frames_000001.tpx3")
