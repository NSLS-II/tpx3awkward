===========
tpx3awkward
===========

.. image:: https://img.shields.io/travis/nsls-ii/tpx3awkward.svg
        :target: https://travis-ci.org/nsls-ii/tpx3awkward

.. image:: https://img.shields.io/pypi/v/tpx3awkward.svg
        :target: https://pypi.python.org/pypi/tpx3awkward


A package for reading tpx3 files to Awkward Arrays

* Free software: 3-clause BSD license
* Documentation: (COMING SOON!) https://nsls-ii.github.io/tpx3awkward.

Features
--------

* TODO


Reference Implementations
-------------------------

* https://github.com/svihra/TimePix3/blob/master/dataprocess.cpp
* https://github.com/emx77/tpx3-to-root/blob/master/tpx3_to_root.cpp

Docs
----

* https://www.quantastro.bnl.gov/node/64

Validation example
------------------

This example validates against the CSV output

.. code:: python


   from pathlib import Path
   import pandas as pd
   from tpx3awkward._utils import raw_as_numpy, ingest_raw_data

   data_dir = Path('/some/path/')

   tmp = []
   for j in range(16):
       csv = data_dir / f'analysis/Test2-68mA-cont/frames_000000-{j}.csv'
       tmp.append(pd.read_csv(csv))
   ref = pd.concat(tmp).reset_index()

   fname = data_dir / 'analysis/Test2-68mA-cont/frames_000000.tpx3'

   d = raw_as_numpy(fname)
   test = pd.DataFrame(ingest_raw_data(d)).reset_index()


   test_sorted = test.sort_values(['timestamp', 'ToT', 'x', 'y'])
   ref_sorted = ref.sort_values(['#ToA', '#ToT[arb]', '#Col', '#Row'])


   assert (test_sorted['x'].values == ref_sorted['#Col'].values).all()
   assert (test_sorted['y'].values == ref_sorted['#Row'].values).all()
   assert (test_sorted['ToT'].values == ref_sorted['#ToT[arb]'].values).all()
   assert (test_sorted['timestamp'].values == ref_sorted['#ToA'].values).all()


The sorting step is needed to compare because the roll-over does not appear to
be correctly handled by either code (which follows because this code has the
exact same logic as the c++ code we were adapting).  However, the c++ code
sorts the photons by ToA in batches.  This means that before and after the roll
over the times are monotonic and the some of the photons in the sort batch will
come out shuffled.

Recreating the "sort by batch" is not worth the trouble, so fully sort things
by ToA, then ToT, then pixels location (to ensure that the sorting is stable,
there are events with identical times).
