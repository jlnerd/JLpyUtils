import sys, os
import pandas as pd
import dask.dataframe

import pytest

import JLpyUtils


tmpdir = 'tmp'

def test_file_utils_save(tmpdir):
    
    JLpyUtils.file_utils.save(pd.DataFrame(),
                              'pandas_foo','csv',tmpdir)
    JLpyUtils.file_utils.save(dask.dataframe.from_pandas(pd.DataFrame(),npartitions=1),
                              'dask_foo','csv',tmpdir)
    JLpyUtils.file_utils.save({},'foo','json',tmpdir)
    JLpyUtils.file_utils.save({},'foo','dill',tmpdir)
    JLpyUtils.file_utils.save(dask.dataframe.from_pandas(pd.DataFrame(),npartitions=1),
                              'dask_foo','h5',tmpdir)
