import sys, os
import numpy as np
import pandas as pd
import dask, dask.dataframe

import pytest

import pyDSlib

import sklearn, sklearn.datasets

def build_data_dict():
    """
    Build a dictionary with different kinds of data to be tested on
    """
    data = sklearn.datasets.fetch_california_housing()
    df_X = pd.DataFrame(data['data'], columns=data['feature_names'])
    df_X['categorical int'] = np.random.randint(0,20, df_X.shape[0])
    df_X['categorical str'] = [['a','b','c'][np.random.randint(0,2,1)[0]] for i in range(df_X.shape[0])]
    df_y = pd.DataFrame(data['target'], columns=['normalized home price'])
    ddf_X = dask.dataframe.from_pandas(df_X, npartitions=3)
    ddf_y = dask.dataframe.from_pandas(df_y, npartitions= 3)
    data_dict = {'df_X':df_X,
                 'df_y':df_y,
                 'ddf_X':ddf_X,
                 'ddf_y':ddf_y}
    return data_dict
    
def test_fetch_color_map_for_primary_color():
    
    arg_grid = [{'primary_color':'R', 'n_colors':3},
                {'primary_color':'G', 'n_colors':3},
                {'primary_color':'B', 'n_colors':3}]
    
    actual = [pyDSlib.plot.fetch_color_map_for_primary_color(**args) for args in arg_grid]
                 
    expected = [np.array([[0.2989711 , 0.        , 0.        , 1.        ],
                        [1.        , 0.09166748, 0.        , 1.        ],
                        [1.        , 0.88431325, 0.        , 1.        ]]),
                np.array([[0.        , 0.6667    , 0.5333    , 1.        ],
                        [0.        , 0.73853137, 0.        , 1.        ],
                        [0.        , 1.        , 0.        , 1.        ]]),
                np.array([[0.        , 0.        , 0.5       , 1.        ],
                        [0.        , 0.09607843, 1.        , 1.        ],
                        [0.        , 0.69215686, 1.        , 1.        ]])]
    message = 'actual value does not match expected value\nactual: {0}\nexpected: {1}'.format(actual,expected)
    
    assert(all([pytest.approx(actual[i])==expected[i] for i in range(len(arg_grid))])), message
    
def test_hist_or_bar():
    data_dict = build_data_dict()

    for key in data_dict.keys():
        
        try:
            pyDSlib.plot.hist_or_bar(data_dict[key], n_plot_columns=min([3,len(data_dict[key].columns)]))
        except:
            print(key)
            raise