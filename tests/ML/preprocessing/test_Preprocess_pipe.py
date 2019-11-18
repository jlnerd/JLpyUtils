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
    X1 = np.linspace(0,1,200)#*np.random.ranf(1000)
    X2 = X1#*np.random.ranf(1000)
    X3 = X1#*np.random.ranf(1000)

    y1 = X2 + 2*X2 + 3*X3
    y2 = X2 - 2*X2 - 3*X3
    
    df_X=pd.DataFrame(np.array([X1,X2,X3]).T,columns=['X1','X2','X3'])

    headers_dict={'continuous features': list(df_X.columns)}
    
    df_X['categorical int'] = list(np.random.randint(0,20, df_X.shape[0]))
    df_X['categorical str'] = [['a','b','c'][np.random.randint(0,2,1)[0]] for i in range(df_X.shape[0])]
    
    headers_dict['categorical features'] = ['categorical int','categorical str']
    
    df_y = pd.DataFrame(y1, columns=['y1'])
    df_yy = pd.DataFrame(np.array([y1,y2]).T,columns=['y1','y2'])

    #build dask equivalent dataframes
    ddf_X = dask.dataframe.from_pandas(df_X, npartitions=3)
    ddf_y = dask.dataframe.from_pandas(df_y, npartitions= 3)
    ddf_yy = dask.dataframe.from_pandas(df_yy, npartitions= 3)
    
    data_dict = {'df_X':df_X,
                 'df_y':df_y,
                 'df_yy':df_yy,
                 'ddf_X':ddf_X,
                 'ddf_y':ddf_y,
                 'ddf_yy':ddf_yy}
    
    return data_dict, headers_dict

def test_PreprocessPipe_with_pandas_df(tmpdir):

    data_dict, headers_dict = build_data_and_headers_dict()

    df_ID = 'df_X'
    PreprocessPipe = JLpyUtils.ML.preprocessing.PreprocessPipe(
                            path_feat_eng_root_dir = tmpdir, overwrite=True)
    PreprocessPipe.fit(data_dict[df_ID], headers_dict)
    PreprocessPipe.transform(data_dict[df_ID])

def test_PreprocessPipe_with_dask_df(tmpdir):

    data_dict, headers_dict = build_data_and_headers_dict()

    df_ID = 'ddf_X'
    PreprocessPipe = JLpyUtils.ML.preprocessing.PreprocessPipe(
                            path_feat_eng_root_dir = tmpdir, overwrite=True)
    PreprocessPipe.fit(data_dict[df_ID], headers_dict)
    PreprocessPipe.transform(data_dict[df_ID])
        
        
        
    
    