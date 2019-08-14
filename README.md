# JLpy_Utilities
Custom modules/classes/methods for various data science, computer vision, and machine learning operations in python

## Dependancies
* General libraries distributed with Anaconda (pandas, numpy, sklearn, scipy, matplotlib, etc.)
* image/video analysis:
    * cv2 (pip install opencv-python)
* ML_models sub-package dependancies:
    * tensorflow or tensorflow-gpu
    * dill
    
## Importing
To import the package, your notebook/code must either be sitting in the same directory as the "JLpy_utils_package" folder, or the package must be in a directory contained in your list of system path (run ```sys.path``` in python/jupyter notebook). Most of the example codes published in my repo. assume you have cloned this repo/package to your desktop, and we simply add the desktop location to your system path via the command:
```sys.path.append(os.path.join(os.path.expanduser("~"),'Desktop'))```
After this, the package can be imported:
```import JLpy_utils_package as JLutils```

## Overview of Package Modules
There are modules in this package:
```JLutils.summary_tables
JLutils.plot
JLutils.img
JLutils.video
JLutils.ML_models
```