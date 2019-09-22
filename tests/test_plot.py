import sys, os
import numpy as np
import pytest

#sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('__file__')),'..')))

import JLpyUtils

def test_fetch_color_map_for_primary_color():
    
    arg_grid = [{'primary_color':'R', 'n_colors':3},
                {'primary_color':'G', 'n_colors':3},
                {'primary_color':'B', 'n_colors':3}]
    
    actual = [JLpyUtils.plot.fetch_color_map_for_primary_color(**args) for args in arg_grid]
                 
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