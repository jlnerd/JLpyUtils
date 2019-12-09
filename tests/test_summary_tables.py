import pytest
import sys, os
import pandas as pd

import pyDSlib

def test_count_subgroups_in_group():
    df = {}
    df['subgroup'] = []
    df['group'] = []
    for color in ['R','G','B']:
        slice_ = [i for i in range(3)]
        df['subgroup'] = df['subgroup']+ slice_+slice_
        df['group'] = df['group'] + [color for vale in slice_+slice_]
    df = pd.DataFrame.from_dict(df)
    
    df_test = pyDSlib.summary_tables.count_subgroups_in_group(df, group_label='group', 
                                                            sub_group_label='subgroup')

    assert(df_test.iloc[0,1]==3), 'expected df_test.iloc[0,1]=3, received df_test.iloc[0,1]='+str(df_test.iloc[0,1])