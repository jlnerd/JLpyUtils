
# coding: utf-8

# In[6]:


#To call in code, add the following command
# import sys, os
# desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
# sys.path.insert(0, desktop_path+'/JLpy_Utilities')
# import ML_Preprocess

#Save the jupyter notebook file as .py to load as module in other notebooks
try:
    get_ipython().system('jupyter nbconvert --to script ML_Preprocess.ipynb')
except:
    print('')

def Scale_Data(df_X,df_y,path_root):
    
    '''
    Description:
        performs min max scaling on X and y dfs and saves the scaler generators to 'Features_Scaler.save' and 'Label_Scaler.save in the 'path_root'
    Inputs:
        df_X, df_y, path_root
    Returns:
        scaler_X, scaler_y, df_X_scaled, df_y_scaled
    '''
    #Load Modules
    from sklearn.externals import joblib
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import pandas as pd
    
    #Build Scalers
    scaler_X = MinMaxScaler()
    scaler_X.fit(np.array(df_X))

    scaler_y = MinMaxScaler()
    scaler_y.fit(np.array(df_y))

    #Save the scalers
    scaler_X_filename = path_root +"\Features_Scaler.save"
    joblib.dump(scaler_X, scaler_X_filename)

    scaler_y_filename =path_root +"\Label_Scaler.save"
    joblib.dump(scaler_y, scaler_y_filename) 
    
    #Scale the data
    df_X_scaled = pd.DataFrame(scaler_X.transform(np.array(df_X)),columns = df_X.columns)
    df_y_scaled = pd.DataFrame(scaler_y.transform(np.array(df_y)),columns = df_y.columns)
    
    # Load Scalers: scaler = joblib.load(scaler_filename) 
    
    return scaler_X, scaler_y, df_X_scaled, df_y_scaled

