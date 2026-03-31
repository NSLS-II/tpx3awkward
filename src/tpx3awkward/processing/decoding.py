from pathlib import Path
from typing import TypeVar

import numba
import numpy as np
import pandas as pd
from numpy.typing import NDArray

IA = NDArray[np.uint64]
UnSigned = TypeVar("UnSigned", IA, np.uint64)


def raw_as_numpy(fpath: str | Path) -> IA:
    """
    Read raw tpx3 data file as a numpy array.

    Each entry is read as a uint8 (64bit unsigned-integer)

    Parameters
    ----------

    """
    return np.fromfile(fpath, dtype="<u8")


@numba.jit(nopython=True, cache=True)
def get_block(v: UnSigned, width: int, shift: int) -> UnSigned:
    return v >> np.uint64(shift) & np.uint64(2**width - 1)


@numba.jit(nopython=True, cache=True)
def matches_nibble(data, nibble) -> numba.boolean:
    return (int(data) >> 60) == nibble


@numba.jit(nopython=True, cache=True)
def is_packet_header(v: UnSigned) -> UnSigned:
    """Identify packet headers by magic number (TPX3 as ascii on lowest 8 bytes]"""
    return get_block(v, 32, 0) == 861425748


@numba.jit(nopython=True, cache=True)
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


@numba.jit(nopython=True, cache=True)
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


@numba.jit(nopython=True, cache=True)
def decode_xy(msg, chip):
    # these names and math are adapted from c++ code
    l_pix_addr = (msg >> np.uint(44)) & np.uint(0xFFFF)
    # This is laid out 16ibts which are 2 interleaved 8 bit unsigned ints
    #  CCCCCCCRRRRRRCRR
    #  |dcol ||spix|^||
    #  | 7   || 6  |1|2
    #
    # The high 7 bits of the column
    # '1111111000000000'
    dcol = (l_pix_addr & np.uint(0xFE00)) >> np.uint(8)
    # The high 6 bits of the row
    # '0000000111111000'
    spix = (l_pix_addr & np.uint(0x01F8)) >> np.uint(1)
    rowcol = _shift_xy(
        chip,
        # add the low 2 bits of the row
        # '0000000000000011'
        spix + (l_pix_addr & np.uint(0x3)),
        # add the low 1 bit of the column
        # '0000000000000100'
        dcol + ((l_pix_addr & np.uint(0x4)) >> np.uint(2)),
    )
    return rowcol[1], rowcol[0]


@numba.jit(nopython=True, cache=True)
def get_spidr(msg):
    return msg & np.uint(0xFFFF)


@numba.jit(nopython=True, cache=True)
def decode_message(msg, chip, heartbeat_time: np.uint64 = 0):
    """Decode TPX3 packages of the second type corresponding to photon events (id'd via 0xB upper nibble)

    Parameters
    ----------
        msg (uint64): tpx3 binary message
        chip (uint8): chip ID, 0..3
        heartbeat_time (uint64):

        # bit position   : ...  44   40   36   32   28   24   20   16   12    8 7  4 3  0
        # 0xFFFFC0000000 :    1111 1111 1111 1111 1100 0000 0000 0000 0000 0000 0000 0000
        # 0x3FFFFFFF     :    0000 0000 0000 0000 0011 1111 1111 1111 1111 1111 1111 1111
        # SPIDR          :                                       ssss ssss ssss ssss ssss
        # ToA            :                                                   tt tttt tttt
        # ToA_coarse     :                          ss ssss ssss ssss ssss sstt tttt tttt
        # pixel_bits     :                          ^^
        # FToA           :                                                           ffff
        # count          :                     ss ssss ssss ssss ssss sstt tttt tttt ffff   (FToA is subtracted)
        # phase          :                                                           pppp
        # 0x10000000     :                           1 0000 0000 0000 0000 0000 0000 0000
        # heartbeat_time :    hhhh hhhh hhhh hhhh hhhh hhhh hhhh hhhh hhhh hhhh hhhh hhhh
        # heartbeat_bits :                          ^^
        # global_time    :    hhhh hhhh hhhh hhss ssss ssss ssss ssss sstt tttt tttt ffff

        # count = (ToA_coarse << np.uint(4)) - FToA     # Counter value, in multiples of 1.5625 ns

    Returns
    ----------
        Arrays of pixel coordinates, ToT, and timestamps.
    """
    msg, heartbeat_time = np.uint64(msg), np.uint64(heartbeat_time)  # Force types
    x, y = decode_xy(msg, chip)  # or use x1, y1 = calculateXY(msg, chip) from the Vendor's code
    # ToA is 14 bits
    ToA = (msg >> np.uint(30)) & np.uint(0x3FFF)
    # ToT is 10 bits; report in ns
    ToT = ((msg >> np.uint(20)) & np.uint(0x3FF)) * 25
    # FToA is 4 bits
    FToA = (msg >> np.uint(16)) & np.uint(0xF)
    # SPIDR time is 16 bits
    SPIDR = np.uint64(get_spidr(msg))

    ToA_coarse = (SPIDR << np.uint(14)) | ToA
    # pixel_bits are the two highest bits of the SPIDR (i.e. the pixelbits range covers 262143 spidr cycles)
    pixel_bits = np.int8((ToA_coarse >> np.uint(28)) & np.uint(0x3))
    # heart_bits are the bits at the same positions in the heartbeat_time
    heart_bits = np.int8((heartbeat_time >> np.uint(28)) & np.uint(0x3))
    # Adjust heartbeat_time based on the difference between heart_bits and pixel_bits
    diff = heart_bits - pixel_bits
    # diff +/-1 occur when pixelbits step up
    # diff +/-3 occur when spidr counter overfills
    # diff can also be 0 -- then nothing happens -- but never +/-2
    if (diff == 1 or diff == -3) and (heartbeat_time > np.uint(0x10000000)):
        heartbeat_time -= np.uint(0x10000000)
    elif diff == -1 or diff == 3:
        heartbeat_time += np.uint(0x10000000)
    # Construct globaltime
    global_time = (heartbeat_time & np.uint(0xFFFFFFFC0000000)) | (ToA_coarse & np.uint(0x3FFFFFFF))
    # Phase correction
    phase = np.uint((x / 2) % 16) or np.uint(16)
    # Construct timestamp with phase correction
    ts = (global_time << np.uint(4)) - FToA + phase

    return x, y, ToT, ts


@numba.jit(nopython=True, cache=True)
def _ingest_raw_data(data):
    chips = np.zeros_like(data, dtype=np.uint8)
    x = np.zeros_like(data, dtype="u2")
    y = np.zeros_like(data, dtype="u2")
    tot = np.zeros_like(data, dtype="u4")
    ts = np.zeros_like(data, dtype="u8")
    heartbeat_lsb = None  # np.uint64(0)
    heartbeat_msb = None  # np.uint64(0)
    heartbeat_time = np.uint64(0)
    hb_init_flag = False  # Indicate when the heartbeat was set for the first time

    photon_count, chip_indx, msg_run_count, expected_msg_count = 0, 0, 0, 0

    for msg in data:
        if is_packet_header(msg):
            # Type 1: packet header (id'd via TPX3 magic number)
            if expected_msg_count != msg_run_count:
                print("Missing messages!", msg)

            # extract the chip number for the following photon events
            chip_indx = np.uint8(get_block(msg, 8, 32))

            # "number of pixels in chunk" is given in bytes not words and means all words in the chunk, not just "photons"
            expected_msg_count = get_block(msg, 16, 48) // 8
            msg_run_count = 0

        elif matches_nibble(msg, 0xB):
            # Type 2: photon event (id'd via 0xB upper nibble)
            chips[photon_count] = chip_indx
            _x, _y, _tot, _ts = decode_message(msg, chip_indx, heartbeat_time=heartbeat_time)
            x[photon_count] = _x
            y[photon_count] = _y
            tot[photon_count] = _tot
            ts[photon_count] = _ts

            # Adjust timestamps that were set before the first heartbeat was received
            if hb_init_flag and (photon_count > 0):
                prev_ts = ts[:photon_count]  # This portion needs to be adjusted
                # Find what the current timestamp would be without global heartbeat
                _, _, _, _ts_0 = decode_message(msg, chip_indx, heartbeat_time=np.uint64(0))
                # Check if there is a SPIDR rollover in the beginning of the file before the heartbeat
                head_max = max(prev_ts[:10])
                tail_min = min(prev_ts[-10:])
                if (head_max > tail_min) and (head_max - tail_min > 2**32):
                    prev_ts[prev_ts < (tail_min + head_max) / 2] += np.uint64(2**34)
                    _ts_0 += np.uint64(2**34)
                ts[:photon_count] = prev_ts + (_ts - _ts_0)

            hb_init_flag = False
            photon_count += 1
            msg_run_count += 1

        elif matches_nibble(msg, 0x6):
            # Type 3: TDC timstamp (id'd via 0x6 upper nibble)
            # TODO: handle these!
            msg_run_count += 1

        elif matches_nibble(msg, 0x4):
            # Type 4: global timestap (id'd via 0x4 upper nibble)
            subheader = (msg >> np.uint(56)) & np.uint64(0x0F)
            if subheader == 0x4:
                # timer LSB, 32 bits of time -- needs to be received first, before MSB
                heartbeat_lsb = (msg >> np.uint(16)) & np.uint64(0xFFFFFFFF)
            elif subheader == 0x5:
                # timer MSB -- only matters if LSB has been received already
                if heartbeat_lsb is not None:
                    if heartbeat_msb is None:
                        hb_init_flag = True
                    heartbeat_msb = ((msg >> np.uint(16)) & np.uint64(0xFFFF)) << np.uint(32)
                    heartbeat_time = heartbeat_msb | heartbeat_lsb
                    # TODO the c++ code has large jump detection, do not understand why
            else:
                raise Exception(f"Unknown subheader {subheader} in the Global Timestamp message")

            msg_run_count += 1

        elif matches_nibble(msg, 0x7):
            # Type 5: "command" data (id'd via 0x7 upper nibble)
            # TODO handle this!
            msg_run_count += 1

        # else:
        # print(f"Exception 'Not supported: {msg}'")
        # raise Exception(f"Not supported: {msg}")

    # Check if there were no heartbeat messages and adjust for potential SPIDR rollovers
    if heartbeat_msb is None:
        # warnings.warn("No heartbeat messages received; decoded timestamps may be inaccurate.")
        # print("No heartbeat messages received; decoded timestamps may be inaccurate.")
        head_max = max(ts[:10])
        tail_min = min(ts[-10:])
        if (head_max > tail_min) and (head_max - tail_min > 2**32):
            ts[ts < (tail_min + head_max) / 2] += np.uint64(2**34)

    # Sort the timestamps
    # is mergesort the best here? wondering if this could be optimized
    indx = np.argsort(ts[:photon_count], kind="mergesort")
    x, y, tot, ts, chips = x[indx], y[indx], tot[indx], ts[indx], chips[indx]

    return x, y, tot, ts, chips


def ingest_raw_data(data: IA) -> dict[str, NDArray]:
    """
    Parse values out of raw timepix3 data stream.

    Parameters
    ----------
    data : NDArray[np.unint64]
        The stream of raw data from the timepix3

    Returns
    -------
    Dict[str, NDArray]
       Keys of x, y, ToT, chip_number
    """
    return {k.strip(): v for k, v in zip(["x", " y", " ToT", " t", " chip"], _ingest_raw_data(data), strict=True)}


def tpx_to_raw_df(fpath: str | Path) -> pd.DataFrame:
    """
    Parses a .tpx3 file and returns the raw data after timesorting.

    Parameters
    ----------
    fpath: Union[str, Path]
        The path to the .tpx3 data to be processed.

    Returns
    -------
    pd.DataFrame
       DataFrame of raw events from the .tpx3 file.
    """
    raw_df = pd.DataFrame(ingest_raw_data(raw_as_numpy(fpath)))
    # should we specify the sorting algorithm? at this point? it should be sorted anyway,
    # but I think dataframes need to be explicitly sorted for use in e.g. merge_asof?
    return raw_df.sort_values(["t", "x", "y", "ToT"]).reset_index(drop=True)
