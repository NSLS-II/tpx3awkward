from __future__ import annotations

import importlib.metadata

import tpx3awkward as m


def test_version():
    assert importlib.metadata.version("tpx3awkward") == m.__version__
