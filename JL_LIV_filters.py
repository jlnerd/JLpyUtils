import numpy as np
import pandas as pd

def filter_by_estimated_IQE(df,group_label = 'Composite_ID',threshold=100):
    df_group = df.groupby(group_label)
    
    print('Unique groups before filter:',len(df[group_label].drop_duplicates()))
    
    df = pd.DataFrame()
    for group_ID, group_subset in df_group:
        if group_subset['Estimated_IQE_prc'].max()<threshold:
            df = pd.concat((df,group_subset))
    
    print('Unique groups after filter:',len(df[group_label].drop_duplicates()))   
          
    return df

