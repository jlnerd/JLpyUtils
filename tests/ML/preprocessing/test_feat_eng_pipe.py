import sys, os
import numpy as np
import pandas as pd
import dask, dask.dataframe

import pytest

import JLpyUtils

def build_data_and_headers_dict():
    """
    Build a dictionary with different kinds of data to be tested on
    """
    import sklearn, sklearn.datasets
    
    data = sklearn.datasets.fetch_california_housing()
    df_X = pd.DataFrame(data['data'], columns=data['feature_names'])
    
    headers_dict={'continuous features': list(df_X.columns)}
    
    df_X['categorical int'] = list(np.random.randint(0,20, df_X.shape[0]))
    df_X['categorical str'] = [['a','b','c'][np.random.randint(0,2,1)[0]] for i in range(df_X.shape[0])]
    
    headers_dict['categorical features'] = ['categorical int','categorical str']
    
    df_y = pd.DataFrame(data['target'], columns=['normalized home price'])
    ddf_X = dask.dataframe.from_pandas(df_X, npartitions=3)
    ddf_y = dask.dataframe.from_pandas(df_y, npartitions= 3)
    
    data_dict = {'df_X':df_X,
                 'df_y':df_y,
                 'ddf_X':ddf_X,
                 'ddf_y':ddf_y}
    
    return data_dict, headers_dict
    
def test_feat_eng_pipe(tmpdir):
    
    data_dict, headers_dict = build_data_and_headers_dict()
    
    for df_ID in ['df_X','ddf_X']:
        feat_eng_pipe = JLpyUtils.ML.preprocessing.feat_eng_pipe(
                                path_report_dir = tmpdir, overwrite=True)
        feat_eng_pipe.fit(data_dict[df_ID], headers_dict)
        feat_eng_pipe.transform(data_dict[df_ID])
        
    
    