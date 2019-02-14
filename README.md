# JLpy_Utilities
Custom General Utility Modules for Python


### Import JLpy_Utilities Modules into python

We assume the module folder is stored on the desktop and has the name 'JLpy_Utilities

`import sys`  
`desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")`  
`path_desktop = os.path.join(os.path.expanduser("~"), "Desktop") # "fastai") #fastai folder or desktop`
`path_utilites = [path_desktop+'/JLpy_Utilities']`
`for path_utility in path_utilites:`
    `if path_utility not in sys.path:`
        `sys.path.insert(0, path_utility)`
`from JLstrings import *`
`from feature_extraction import *`
`from ML_Preprocess import *`
`from JLplots import *`
