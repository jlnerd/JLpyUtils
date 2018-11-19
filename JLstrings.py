
# coding: utf-8

# In[2]:


#To call in code, add the following command
# import sys, os
# desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
# sys.path.insert(0, desktop_path+'/JLpy_Utilities')
# import JLstrings

#Save the jupyter notebook file as .py to load as module in other notebooks
try:
    get_ipython().system('jupyter nbconvert --to script JLstrings.ipynb')
except:
    print('')

#Load necessary modules
import pandas as pd

#Define Module Functions
def standardize_headers(df):
    #docstring
    """
    replace illegal characters in the df headers with '' or some legal variant of the character (i.e '&' is replaced by 'and')
    """
    
    headers_original = pd.DataFrame(df.columns,columns=['Original_Headers'])
    
    illegal_chars = [' ','{', '}', '?', '$', '%', '^', '&', '*', '(', ')','-','#', '?',',','<','>', '/', '|', '[' ,']','@','λ']
    for char in illegal_chars:
        if char == ' ':
            df.columns = df.columns.str.replace(char,'_')
        elif char == '#':
            df.columns = df.columns.str.replace(char,'num')
        elif char == '&':
            df.columns = df.columns.str.replace(char,'and')
        elif char == '|':
            df.columns = df.columns.str.replace(char,'abs')
        elif char == '@':
            df.columns = df.columns.str.replace(char,'at')
        elif char == 'λ':
            df.columns = df.columns.str.replace(char,'lambda')
        else:
            df.columns = df.columns.str.replace(char,'')
            
    headers_standardized = pd.DataFrame(df.columns,columns=['Standardized_Headers'])
    
    headers_2Darray = pd.concat((headers_original,headers_standardized),axis = 1)
    
    df = pd.DataFrame(df)
    
    return df, headers_standardized, headers_original, headers_2Darray
    

