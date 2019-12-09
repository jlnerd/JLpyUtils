import sys, os
import pandas as pd
import dask.dataframe

import pytest

import pyDSlib


tmpdir = 'tmp'

def test_file_utils_save(tmpdir):
    
    pyDSlib.file_utils.save(pd.DataFrame(),
                              'pandas_foo','csv',tmpdir)
    pyDSlib.file_utils.save(dask.dataframe.from_pandas(pd.DataFrame(),npartitions=1),
                              'dask_foo','csv',tmpdir)
    pyDSlib.file_utils.save({},'foo','json',tmpdir)
    pyDSlib.file_utils.save({},'foo','dill',tmpdir)
    pyDSlib.file_utils.save(dask.dataframe.from_pandas(pd.DataFrame(),npartitions=1),
                              'dask_foo','h5',tmpdir)
